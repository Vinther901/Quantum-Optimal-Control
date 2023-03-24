# Quantum-Optimal-Control

The original setup of this project, was to have the legwork of the code contained in the folder `scripts`

With all the control functions contained in `Controls.py` and the time evolution relevant methods in `Evolvers.py`

`Systems.py` was to setup a model of the physical system and within `Training.py` was all the different loss functions and update methods.

This would all be run in `2DSystems -> RunExperiments.ipynb` and everything except the large data files were saved in `Experiments`


However, once i realized ODESolvers was the way to go I shifted to Julia with which I had no experience. From there i devolved into just creating new notebooks all the time, which was fine since I had a lot of experimentation to do and this was the quickest way. But it means that the rest of the notebook will be difficult to navigate for anobody but the author (sometimes including the author..).

That being said, all the relevant stuff should be contained in `Production` with `Production -> RobustPhiExtContinuousLoss -> RobustPhiExtOptExactCont.ipynb` containing one of the most general adjoint method implementations, within this repo.
