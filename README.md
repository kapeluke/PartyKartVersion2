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

Steps to download/install:
1. Download the GitHub repository
1. Install Python version 3.14.4
1. Install the required packages: `pip3 install -r requirements.txt`
1. Run the application by running the following command in your terminal (within the same directory/folder as main.py): `python3 -m uvicorn main:app --reload --port 5000`. Note: You can change the port if you wish. Just make sure you use the same number in the next step.
1. Load the web app in your browser by loading the page: `http://127.0.0.1:5000` (or whatever the uvicorn output specifies in its start up)

Steps to operate:
1. Enter the number of players that will be racing with the slider and type in their names
1. Enter the number of races each player will be racing with the slider
1. _Optionally_ tweak the generation algorithm by going to the _Tweaks_ section on the bottom left. Here you can edit any hyperparameters, seeds, timeouts, and other values that affect how the AI and algorithm generate the race schedule
1. _Optionally_ enter some race **modifiers** by visiting the _Modifiers_ tab. Modifiers are shuffled at random when generating all the races. Each _channel_ is shuffled and drawn from when a race is starting. So if each race has three modifiers (for example), you have three distinct channels
1. Hit _Start Engine!_ to generate the race schedule. This may take anywhere from 5 to 60+ seconds depending on your choice of the number of players, the number of races for each player, and other hyperparameters
1. Once that's done, the program should show you the current race roster front and center; the leaderboard on the left side; and all the next _on deck_ races on the right
1. You can view the current race's modifiers (if any) by clicking _Modifiers_, and then you can go back again if you wish
1. Enter each player's position after finishing the race and hit _Next Race_ to update the leaderboard and see the next race and their modifiers
1. If at any point you made an error or wish to go back to the previous race, I have a convenient _Undo_ button you may press
1. **FUTURE:** You may _Add_ and _Drop_ players at any time after any number of races have completed and re-generate the upcoming races with the change applied. This will rebalance and redistribute your racers accordingly. This may take some time

## Future/In progress work
This project is not complete yet, and there's a few things remaining:
1. The ability to _Add_ and _Drop_ players in the middle of the tournament. This problem is complex and requires significant effort. Should be done it soon
1. The _Play Again_ button at the end screen does not work. It should bring you back to the home screen or first race
1. Add a way for the user to return to the home screen in the application, other than just closing it down and reopening it
1. Compile it all into a neat executable rather than forcing the user to install everything manually. The user should be able to open it as a desktop application from a `.exe` file rather than running a command in their terminal and needing to visit their browser

## How does it work?
The problem of distributing racers into a fair schedule is really a combination of two problems:

### 1. Race generation
This part tackles the first question:
> How can we distribute $P\in\mathbb{N}$ players, each racing $N\in\mathbb{N}$ times (exactly), into races of size 3 (inclusive) to 4 (inclusive).

The reason we want races to be exactly either size 3 or 4 is really because our tournament/competition would be pretty boring only watching two people go head-to-head or watching someone race by themselves. You might say, _"well, why not 5 or more people in a race?"_ Fair question. The real answer is because of 'number of controller' limitations when designing the problem. Maybe in the future I will allow you, the user, to edit the sizes of the races when generating the races. But for now, it sticks to sizes of 3 or 4.

Imagine $P$ players sitting (evenly distributed) around a circle. Now, let's asume the player at the twelve o'clock position is always chosen — they'll be the player at index $0$. To choose the remaining 3 people in the race, you need to choose three _offsets_ from index $0$. So to choose players: 0, 4, 6, 7 to race, our offets are then $\langle4,2,1\rangle$. We call the choice of three offsets to be called a _stencil_. Applying the stencil to our roster, we get the players in the race: $\langle0+4,=+4+2,0+4+2+1\rangle=\langle0,4,6,7\rangle$. We call these _holes_, akin to imaginging a piece of cardboard placed over top of a $P$-sided shape and cutting holes out overtop of players 0, 4, 6, and 7.

To generate a valid race schedule, we can actually just rotate our holes around the circle, like rotating a piece of cardboard. So our next races will still have the same offsets, but will be: $\langle1,5,7,8\rangle$, $\langle2,6,8,9\rangle$, $\ldots$ Now, you may notice that we may have $P<9$, so every hole is modulus $P$. This generates $P$ number of races (since we can only add 1, modulus $P$, $P$ times before getting duplicates).

Unfortunately, this doesn't make every player races $N$ times. We at least need $\lceil\frac{NP}{4}\rceil$ races to do that, so we use multiple stencils in a _superpool_ as candidate choices for our algorithm to pick from. Because we want the algorithm to be flexible to allow for 3-person races, we generate these by 'patching' some _holes_ to create 3-person races instead. So in our example of $\langle0,4,6,7\rangle$, we may choose to make it $\langle4,6,7\rangle$ or $\langle0,6,7\rangle$ or $\langle0,4,7\rangle$ or $\langle0,4,6\rangle$. The number of 3-person races taken from every set of _holes_ is determined via hyperparameters: _Cuts per stencil rotation_. Maximum 4, minimum 0.

We then have a list of candidate races, and the problem is to choose them to minimize the player-to-player race count variance. For choice of $R$ races, we have an _incidence matrix_, $A$. This is a matrix of 1's and 0's, of size $\text{rows}=R,\text{cols}=P$. So if $0\le i\lt P$, $0\le j\lt R$, then if $A_{i,j}=1$, it means player $i$ is racing in race $j$. Every column in the incidence matrix will sum to 3 or 4, and every row will sum to exactly $N$.

To find how many times player $a$ has raced against a player $b$, we construct the _co-occurence matrix_, $C=AA^\top$. Every entry in the matrix $C_{a,b}$ will be the count of every race where player $a$ and $b$ have both been a part of. Thus, $C_{a,b}=C_{b,a}$ . Every diagonal element, $C_{a,a}$ should be exactly $N$.

And now finally we have our method of valuing a selection of candidates. We pass in the selection of candidates into the _Integer Linear Programming_ (ILP) solver, and tell it to make an incidence matrix out of our list of candidates, where each row sums to $N$. We rank its solutions based on the variance of the non-diagonal elements of $C$, and punish higher variance solutions with a higher loss. Because the ILP is deterministic, we shuffle the candidates with a provided seed and try different superpools to see what gets the best solution. We then return the solution with the best score (lowest loss).

### 2. Race scheduling
Now that we have our incidence matrix, $A$, which is our list of unordered races, we can now ask the question:
> How do we order the races (columns in $A$) such that every player has their races spread out as evenly as possible?

We don't want a player to race all of their races at the begining and then have a super long wait until the end of the tournament to race their final race. That's boring to have to sit and wait (and if you're making it a drinking game, it is just not fun to be super drunk then sober up).

There's a couple ways to tackle this problem:
1. Using nearly the same technique as selection the races: _Mixed-Integer Linear Programming_ (ILP). This provides mathematically very sound constructions, but can take a very long time for a medium to large number of races, $R>15$ (we're talking hours to days). 
2. Using _Simulated Annealing_. We will describe this technique in further detail below. This does not produce as nice of results, but is exceptionally faster for large values of $R$.

So, we use both. For any $R\le15$ we use the MILP solver, and for any $R>15$ we use simulated annealing. For both ojectives, we want the races that a player participates in to be spread apart by approximately $\mu=\frac{R-1}{N-1}$. The loss is then the sum of squared differences of each 'gap' from $\mu$.

I won't go over the MILP solver in detail, but you're welcome to check out `race_ordering.py`.

Simulated Annealing is the method chosen to solve the larger ($R>15$) problems. We perform random swaps (according to a seed) of two races' positions in the schedule, and ask whether it reduces the overall loss. If it does, we take the swap, otherwise we don't. However, because this can enter local optima and get stuck (rather than reaching the more desirable global optimum), we want to introduce randomness into the equation. With some probability we take a swap even if it makes the loss higher. This probability is: $p(s|o)=\exp\left(\frac{E(o)-E(s)}{\tau+\varepsilon}\right)$, where $s$ is the proposed ordering after the swap, $o$ is the original ordering, $\tau$ is the _temperature_, and $\varepsilon$ is a small constant to prevent division by 0. $E$ is the function which determines the loss of the proposed ordering. In the beginning, we want a higher _entropy_, so we set _temperature_ to be higher, and thus we take more random moves. As the simulated annealing process continutes, we linearly lower the value. This occurs for some number of steps, until at 80% of the way through the process, we set $\tau=0$, and only take greedy moves for the remaining 20% of the steps.

Both processes produce an ordered incidence matrix, which we then use as our race schedule.

### 3. Modifiers
Now I know I said there's only two problems, but I meant two _major_ problems. To make it more special than just a race scheduler, we add modifiers into the mix. These are done very easily. Suppose we have $R$ races, we want to choose $R$ modifiers from a set or _channel_, such that it's one per race per channel. So if you want your races to have 3 modifiers each, you need 3 channels. For every race, we draw one modifier from each channel.

Rather than drawing modifiers from each channel, we shuffle the channels. That way, every race has a different modifier and we get a chance to see all of them in action. There's a chance that the size of the channel is less than $R$, so in that case we will see duplicates. In that case, I will reuse modifiers, but they will be randomly shuffled again.

### 4. Front end UI
Yes, there's another problem, but this is another small necessary one: making the visualizations and control over the algorithm. For quick details, the UI is built using the [FastAPI](https://fastapi.tiangolo.com/) framework, in combination with [HTMX](https://htmx.org/), [Tailwind CSS](https://tailwindcss.com/), CSS, and JavaScript. Some icons come from [FontAwesome](https://fontawesome.com/icons). I'm not much of a front end type of person, so [Google Gemini](https://gemini.google.com/app) was a good help putting together a good looking UI.

## What do the _Tweaks_ do?
The _Tweaks_ is essentially the settings of the solver. Please read below for detailed information on each tweak:
1. **'strict_R' (boolean):** Whether we want to toggle on the _strict_ version of race selection. By turning this on, you force the system to choose _exactly_ $\lceil\frac{NP}{4}\rceil$ races. This can make the solution take much longer to find; or even potentially impossible.
1. **'strictness_tolerance' (integer):** Only matters when **strict_R** is set to false. This sets the upper limit for the number of races $R$. The upper limit is then $\lceil\frac{NP}{4}\rceil+\text{strictness tolerance}$. The lower the number, the longer the algorithm may take to solve the problem.
1. **'cuts_per_rotation' (integer):** How many holes of size 3 to create from each holes of size 4. Must be at minimum 0 (inclusive) and at most 4 (inclusive). You will get better results with a higher number, but it may take longer to solve the problem.
1. **'num_stencils' (integer):** The number of stencils to consider choosing from. Higher is better, but increases the potential length of solving.
1. **'pool_size' (integer):** The size of our superpools. We take the **num_stencils** and divide them into pools of this size. This creates the number of candidates for the ILP solver.
1. **'num_restarts' (integer):** The number of random tries that each pool of candidates gets. The ILP solver is essentially deterministic, so for each restart we randomly permute the candidates to yield a different solution.
1. **'seed' (integer):** The random seed used for everything. Different seeds yield different results whether it's the same $P$ and $N$ and other hyperparameters. 
1. **'gap_variance_weight' (float):** How much weight we want to give to the offsets in each stencil. We typically want the stencil holes to be evenly spaced to improve the solutions. We don't want this to be too high though, or we place too much emphasis on it over other important metrics. This is by default set to $0.1$.
1. **'num_climbers' (integer):** The number of parallel actors attempting to solve the simulated annealing problem of race ordering. More is better, but slows the solution down.
1. **'num_steps' (integer):** The number of steps (swaps) each climber performs in total (with decreasing temperature) to attempt to properly order the race schedule. Higher is better, but slows the algorithm down.
1. **'ilp_timeout' (float):** How long should the ILP solver in race selection search before declaring it can't find something and tossing out its solution? Generally this shouldn't matter because the ILP solver shouldn't time out often, but it's important to have just in case. 5 seconds is sufficient here.
1. **'milp_timeout' (float):** How long should the MILP solver in race ordering search before declaring it can't find something and tossing out is solution? We only use the MILP when the number of races is $\le15$, so realistically this shouldn't take a very long time anyways, but I give the option for you here. 60 seconds should be sufficient for most $R\le15$ problems.