import yaml
import optuna
from typing import Optional
import os

class YMLstudy():
    '''
    Homemade Class for to save studies as YML file.
    Display and save Optuna Studies in the folder optuna-study. It creates one if does not exist
    Parameters
    ----------
    model [str]: name of model you are working on

    Returns
    ----------
    None
    '''
    
    def __init__(self, model:str='TransChess'):

        os.makedirs("optuna-study", exist_ok=True)

        self.model = model
        self.file_name = f'{model}-opt_parm.yml'
        self.location = f'optuna-study/{self.file_name}'
        
        if os.path.isfile(self.location):
            self._get_vals()
    
    def read(self, what:Optional[str] = None):

            if what in ['all', None]:
                return self.best_all 
            else:
                return self.best_all[what]


    def write_study(self, study: optuna.Study):
        """
        Save only best_trials to YAML file
        or the efficient frontier in case of multi-optim
        """
        with open(self.location, "w") as writer:
            pareto_data = []
            for t in study.best_trials:
                pareto_data.append({
                    "trial_number": t.number,
                    "values": [round(v, 4) for v in (
                        t.values if isinstance(t.values, (list, tuple)) else [t.value]
                    )],
                    "params": t.params,
                })

            yaml.safe_dump(
                {"best_trials": pareto_data},
                writer,
                default_flow_style=False,
                sort_keys=False,
            )

        
    def write_best_param(self,file:str = 'opt-configs.yml'):

        self._get_vals()

        with open(file, "w") as writer:

            yaml.safe_dump(
                self.best_parameters,
                writer,
                default_flow_style=False,
                sort_keys=False,
            )


    def _get_vals(self):
        """
        Load the first best trial from YAML (for quick access).
        the best values will be those in the parato frontier that
        gives the minimum.
        """
        with open(self.location, "r") as loader:
            out = yaml.safe_load(loader)
            self.best_all = out

            best_trials = out.get("best_trials", [])

            if not best_trials:
                self.best_idx = None
                self.best_parameters = None
                self.results = None
                return None

            best_trial = min(best_trials, key=lambda t: 
                0.7*t["values"][0]+0.3*t["values"][1])

            self.best_idx = best_trial["trial_number"]
            self.best_parameters = best_trial["params"]
            self.results = best_trial["values"]
            self.results_sum = sum(best_trial["values"])