import os
import traceback

import pandas as pd

from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

class Experiments(object):
    """docstring for Experiments."""

    def __init__(self, path):
        super(Experiments, self).__init__()
        self.path = path
        self.dict_of_data = self.tflog2pandas()

    # Extraction function
    def tflog2pandas(self):
        runlog_data = {}#pd.DataFrame({"metric": [], "value": [], "step": []})
        try:
            event_acc = EventAccumulator(self.path)
            event_acc.Reload()
            tags = event_acc.Tags()["scalars"]
            for tag in tags:
                event_list = event_acc.Scalars(tag)
                values = list(map(lambda x: x.value, event_list))
                step = list(map(lambda x: x.step, event_list))
                r = {"value": values, "step": step}
                r = pd.DataFrame(r)
                runlog_data[tag] = r
                #runlog_data = pd.concat([runlog_data, r])
        # Dirty catch of DataLossError
        except Exception:
            print("Event file possibly corrupt: {}".format(self.path))
            traceback.print_exc()
        return runlog_data


def export_metrics(path, metrics):

    runlog_data = {}#pd.DataFrame({"metric": [], "value": [], "step": []})
    try:
        event_acc = EventAccumulator(path)
        event_acc.Reload()
        tags = event_acc.Tags()["scalars"]
        for tag in tags:
            if tag not in metrics: continue
            event_list = event_acc.Scalars(tag)
            values = list(map(lambda x: x.value, event_list))
            step = list(map(lambda x: x.step, event_list))
            r = {"value": values, "step": step}
            r = pd.DataFrame(r)
            runlog_data[tag] = r
            #runlog_data = pd.concat([runlog_data, r])
    # Dirty catch of DataLossError
    except Exception:
        print("Event file possibly corrupt: {}".format(path))
        traceback.print_exc()
    return runlog_data
