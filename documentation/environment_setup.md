# Environment setup

## Environment creation
The conda environment necessary for working with the project is defined via [environment.yml](./../environment.yml).
In order to create a local environment using conda run this command in the project's root folder:

`> conda env create --file=environment.yml` 

Additionally, in order to access the local helper functions in the notebooks you should install a local [question_answering](./../src/question_answering) package. To do that run the following command after creating the project's environment:

`> conda activate question_answering` to activate the project's environment

`> cd src` to move to the local package directory

`> pip install -e .` to install the package in development mode.

## Initial jupyter setup
To ensure that the environment's kernels are available to jupyter it is necessary to run this command once:

`> python -m ipykernel install --user --name question_answering`

## Environment updates
In order to update the environment add an appropriate dependency in the [environment.yml file](./../environment.yml) and run this command in the project's root folder:

`> conda env update -f environment.yml` 

## Conda optimization
With the classic conda dependency solver the time needed to solve the dependencies for this project may take a very long time. In order to reduce this time to a few minutes one, one can install ***libmamba-solver***, which is an optimized version of the classic conda solver.

To install ***libmamba-solver*** run:

`> conda install -n base conda-libmamba-solver`

`> conda config --set solver libmamba`

After performing these steps creating and updating the  conda environment should be way quicker.