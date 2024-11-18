import numpy as np
from matplotlib import rcParams
from matplotlib.axes import Axes

from dashboard.widgets import Widget
from i2mb.utils import global_time


class BaseText(Widget):
    text_ = ""

    def __init__(self, ax: Axes, population, experiment_properties=None, intervention=None, title=None):
        self.experiment_properties = {"trace_contacts": None, "quarantine_households": False,
                                      "test_to_exit": False, "night_out": True}
        if experiment_properties is None:
            experiment_properties = {}

        self.experiment_properties.update(experiment_properties)
        self.intervention = intervention
        self.population = population
        self.ax = ax
        self.ax.set_title(title, loc="left")
        self.ax.set_axis_off()
        self.stats_text = self.ax.text(0.0, 0.0, type(self).text_, wrap=True,
                                       # animated=True,
                                       transform=ax.transAxes,
                                       # clip_on=False,
                                       verticalalignment="bottom",
                                       animated=True)


class StaticText(BaseText):
    text_ = """Contact tracing strategy: {ct}
DCT compliance rate: {dor:0.0f}%
Quarantine contact's household (QCH): {hi}
Request negative test to exit quarantine (RNT): {te}
Closure of bars and restaurants (CBR): {cbr}"""

    def __init__(self, ax: Axes, population, experiment_properties=None, intervention=None, title=None):
        super().__init__(ax, population, experiment_properties, intervention, title=title)
        ct = self.experiment_properties["trace_contacts"]
        ct = "ICT, MCT, DCT" if ct == "both" else ct
        self.stats_text.set_text(StaticText.text_.format(dor=(1 - self.experiment_properties["dropout"]) * 100,
                                                         ct=ct,
                                                         hi=self.experiment_properties["quarantine_households"],
                                                         te=self.experiment_properties["test_to_exit"],
                                                         cbr=not self.experiment_properties["night_out"]))
        self.stats_text.set_animated(False)

    def update(self, frame):
        return []


class DynamicText(BaseText):
    text_ = """
    Pathogen:        
        Clearance period:
        Incubation period:
        Generation interval:
        Serial interval:        
    Population:
        7-day hosp. incidence:
        7-day incidence:
        Affected population:
    Agent:    
        Avg. Days in quarantine:
        Avg. Days in isolation:
        Avg. confinements: {anip:0.1f}
        False positive rate: {fp:0.1f}%
        False negative rate:
        Currently confined: {isolated}"""
    locations_ = ["home", "office", "bar", "restaurant", 'bus', 'car']

    def update(self, frame):
        if hasattr(self.population, "location_contracted"):
            cpl = self.population.location_contracted == DynamicText.locations_
        else:
            cpl = np.zeros((len(self.population), 1))

        fp = anip = isolated = 0
        if self.intervention is not None:
            num_isolations = self.intervention.num_confinements[self.intervention.num_confinements > 0]
            if len(num_isolations):
                anip = np.mean(num_isolations)
                fp = self.intervention.isolated_fp.sum() / sum(num_isolations) * 100

            isolated = np.sum(self.intervention.isolated)

        self.stats_text.set_text(DynamicText.text_.format(
            fp=fp,
            anip=anip,
            isolated=isolated))
        return self.stats_text,


class SimClock(BaseText):
    text_ = """{time}"""

    def __init__(self, ax: Axes, population, experiment_properties=None, intervention=None):
        super().__init__(ax, population, experiment_properties, intervention)
        self.fixed_text = ax.text(x=0.5, y=0.0, s="Elapsed time [Days - HH:MM]", ha="center", va="bottom",
                                  transform=ax.transAxes)
        y_base = self.fixed_text.get_transform().inverted().transform_bbox(self.fixed_text.get_tightbbox()).y1

        self.stats_text.set_text("0D-00:00")
        self.stats_text.set(fontsize=rcParams["font.size"] * 5, fontfamily="Ubuntu Mono", ha="center",
                            x=0.5, y=y_base, va="bottom", animated=True)

    def update(self, frame):
        hour = global_time.hour(frame)
        minute = global_time.minute(frame)
        day = global_time.days(frame)

        self.stats_text.set_text(SimClock.text_.format(time=f"{day:3}D-{hour:02d}:{minute:02d}"))
        return self.stats_text,


class Table(Widget):
    def __init__(self, ax: Axes, row_names: list | list[list], row_value_formats: list | list[list] = None,
                 section_titles: list = None,
                 titles_properties=None, names_properties=None, value_properties=None, title=None):

        self.ax = ax
        self.ax.set_title(title, loc="left")
        self.ax.set_axis_off()

        self.value_properties = {"ha": "left"}
        self.value_properties.update({} if value_properties is None else value_properties)

        self.names_properties = {"ha": "right"}
        self.names_properties.update({} if names_properties is None else names_properties)

        self.titles_properties = {"fontweight": "bold", "ha": "center"}
        self.titles_properties.update({} if titles_properties is None else titles_properties)

        self.section_titles = [] if section_titles is None else section_titles
        self.row_names = row_names

        self.section_titles_index = []
        self.row_names_index = []

        label_column = self.fill_label_column()
        text_matrix = np.full((len(label_column), 2), "", dtype=object)
        text_matrix[:, 0] = label_column
        self.table = ax.table(text_matrix, bbox=[0, 0, 1, 1], animated=False,
                              edges="open",
                              )
        self.format_table()
        self.row_value_formats = (row_value_formats if row_value_formats is not None
                                  else ["{0:04d}" for r in self.row_names_index])
        self.value_texts = self.get_value_texts()
        self.__values = np.empty(len(self.value_texts), dtype=object)

    def fill_label_column(self):
        label_column = []
        if self.section_titles:
            section_title_index = 0
            for st, sls in zip(self.section_titles, self.row_names):
                label_column.append(st)
                label_column.extend(sls)
                self.section_titles_index.append(section_title_index)
                self.row_names_index.extend(range(section_title_index+1, section_title_index + 1 + len(sls)))
                section_title_index += 1 + len(sls)

        else:
            label_column = self.row_names

        return label_column

    def format_table(self):
        self.table.auto_set_font_size(False)
        for c in self.table.get_celld().values():
            c.PAD = 0

        for index in self.section_titles_index:
            c = self.table.get_celld()[index, 0]
            c.get_text().set(**self.titles_properties)

            # # Merge title cell
            # c2 = self.table.get_celld()[index, 1]
            # c.get_text().set_bbox(bbox=dict(width=1))

            # c2.set_width(0)

        for index in self.row_names_index:
            c = self.table.get_celld()[index, 0]
            c.get_text().set(**self.names_properties)

            vc = self.table.get_celld()[index, 1]
            vc_text = vc.get_text()

            # Connect the text to the axes, so that we can handle the text only instead of the entire cell.
            vc_text.axes = self.ax
            vc_text.set(**self.value_properties)
            vc.set_animated(True)

    def get_value_texts(self):
        value_text = []
        for index in self.row_names_index:
            vc = self.table.get_celld()[index, 1]
            value_text.append(vc.get_text())

        return value_text

    def update(self, frame):
        for text, text_format, value in zip(self.value_texts, self.row_value_formats, self.__values):
            text.set_text(text_format.format(value))

        return self.value_texts

    def update_values(self, values):
        if len(values) != len(self.__values):
            raise RuntimeError("values needs to have the same number of elements than row_names.")

        self.__values[:] = values




