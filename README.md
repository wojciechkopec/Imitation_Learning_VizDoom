# Imitation_Learning_VizDoom
Learning from visual input with tensorflow and VizDoom environment.
<br>
Firstly, make sure that you have VizDoom installed (See VizDoom page https://github.com/Marqt/ViZDoom for details)
<br>
Then, run "python spectator.py health_gathering result.pkl 1" to collect training trajectories from expert (that's you!).
<br>
Finally, let agent learn from your example: run "python experiment_config.py health_gathering 3000 result.pkl", wait for a while and observe!
<br>
Checked with python 2.7.12, tensorflow 1.0.1 and VizDoom 1.1.0rc1




