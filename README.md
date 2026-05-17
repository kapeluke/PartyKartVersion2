# Party Kart (Version 2)

Welcome to Version 2 of my Party Kart application! This version is much cleaner, faster, and easier to use!

## What can I use this application for?
If you've got a game (like Party Kart) where you require scheduing a specific number of individuals into a very specific number of games fairly and evenly, then this application is for you!

Upon loading the app, you will be presented a screen where you enter the player roster and specify the number of races each player must race in. Then by hitting _Start Engine_ the program will generate a schedule of races for you.

The schedule generated will emphasize: 
1. Lowering the player-to-player race number variance (i.e. making it so every player races against every other player as evenly as possible)
1. Spreading every player's races out as evenly as possible to give every player adequate breaks without making your tournament boring

## What is Party Kart?
***Party Kart*** is a version of Mario Kart (copyright Nintendo), or any other of your favourite kart racing game, where each player is given a set of tasks to complete during the race before they cross the finish line. Typically, the rules might look like:
- You may not do any two tasks at the same time and you cannot drive (in game) while completing any task
- Each player must complete an alcoholic beverage in its entirety before crossing the finish line
- You must complete any other tasks randomly drawn from a set of _modifiers_

Of course, **you do not need to do any of these rules** and you maye just find yourself needing an application to schedule races/games fairly amongst a set of people needing to compete in a specific and exact number of races/games. **This application still works for that!**

## How do I use this application?
**Note, this application is still in development. Because of this, the program is not yet distributed as an executable (.exe) file.**

Steps:
1. Download the GitHub repository
1. Install Python version 3.14.4
1. Install the required packages: `pip3 install requirements.txt`
1. Run the application: `python3 -m uvicorn main:app --reload --port 5000`
1. Load the web app in your browser by loading the page: `http://127.0.0.1:5000` (or whatever the uvicorn output specifies in its start up)
