squad:
Ulpipi + support + medic

=== battle ===
- Symbiosis
    | one line, all ground enemy, ez
    | Emergency == No
- Cistern
    | one line, all ground enemy, rng ez
    | Emergency == Yes
- Insect Infestation
    | one line, all ground enemy, rng ez
    | Emergency == Yes
- Mutual Aid
    | one line, mixed ground + range enemy, rng medium, -2 lifepoint
    | Emergency == No
- Sniper Squad
    | one line, all ground enemy, rng ez
    | Emergency == Yes


=== encounter ===
Inheritance
    | 1 chooice
    | obtain 1 random collectible
Devouring Dust
    | 1 chooice
    | obtain 1 collectible
Cliffside Burial
    | 1 chooice
    | obtain 1 collectible
Seaborn Scholar
    | 2 chooice
    | select the first one to obtain 1 collectible
Puppy-Dog Eyes
    | 2 chooice
    | select second to skip
Homecoming
    | 3 chooice
    | select third to skip
Catastrophe Messenger
    | 2 chooice
    | select first to obtain 1 collectible
Delusions of Lunacy
    | 3 chooice
    | select third to skip


=== second floor ===
Gathering Stormclouds
    | 1 chooice
    | obtain 1 collectible
The Last Tidewatcher
    | 3 chooice
    | select first to obtain 1 random collectible
Path of Suffering
    | 2 chooice
    | select second to skip
Overseas Export
    | 2 chooice
    | select second to skip
Ocean's Legacy
    | 2 chooice
    | select first to obtain 2 random collectible
Faith
    | 2 chooice
    | select second to skip
Chance Meeting
    | 3 chooice
    | select third to skip
Medic's Will
    | 2 chooice
    | select second to skip
Booty Bay
    | 2 chooice
    | select second to skip
Camp
    | 1 chooice
    | obtain 1 collectible


# step by step migrate
- crop all encounter image, each IS have different encounter but mostly there's encounter, combat, emergency and downtime_recreation
- we split each function and step like this :
1. start journey
    - we select which squad, initial requirement and our solo lane operator
2. select stage
    - identify active stage
    - if combat/emergency we auto deploy them
        - only pick the first box of operator and the tile to deploy them
3. end stage
4. automate rogue trader
5. end journey
