# dPickleBallEnv [https://dpickleball.github.io/](https://dpickleball.github.io/)

![image](poster.png)


# Installation Steps to Set Up dPickleball Environment:

1) conda create -n dpickleball python=3.10.12
2) conda activate dpickleball
3) git clone https://github.com/dPickleball/dpickleball-ml-agents.git
4) cd dpickleball-ml-agents
5) pip install -e ./ml-agents-envs
6) pip install -e ./ml-agents
7) pip install matplotlib
8) pip install opencv-python
9) Download this repo, modify the path in test_paral.py

Build for Windows/MacOS/Linux [https://drive.google.com/drive/folders/1LiFp2MJdwzjgP88rJtZO4qPkyfAGSS7e?usp=sharing](https://drive.google.com/drive/folders/1LiFp2MJdwzjgP88rJtZO4qPkyfAGSS7e?usp=sharing)

# Usage:
1) Download the build from the URL above
2) conda activate dpickleball
3) python test_paral.py    (**remember to change path and point to the build**)
4) Play at Unity

# Rules:
1) Agents can move freely within their own half of the court. Paddles can rotate to control the ball’s rebound angle and direction, allowing for angled shots, spin, and advanced control.
2) Your agent takes real-time visual observation of the environment as input and outputs actions. The action space consists of three discrete components, each with three possible values: (0: none, 1: up, 2: down), (0: none, 1: right, 2: left), and (0: none, 1: counter-clockwise, 2: clockwise).
3) The objective is to hit the ball to the opponent’s side of the court. One point is awarded if the opponent fails to return the ball. The first player to reach 21 points wins the match.
4) The winner of the previous point serves the next ball. At the beginning of the match, the right-side player serves first.
5) If the ball becomes stuck in the center of the court due to simultaneous contact by both players, it is considered a held ball. In such cases, the ball will be relocated to the server’s side for a proper re-serve.
6) If the ball remains on one side of the court for more than 5 seconds without crossing over, the opposing player is awarded a point.


# Developed with love and heart by the dPickleball Technical Team
