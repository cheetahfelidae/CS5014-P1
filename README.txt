### README File ###

Usage:
    data_preparation.py
        - there is no required commandline argument
        - but the programme needs the data file name "energydata_complete.csv" at the same directory as the file
        - the programmes produces four files: "training_x.txt", "training_y.txt", "test_x.txt" and "test_y.txt".

    model_trainer.py
        - there is only one required commandline argument: the name desired model as follow
            '-lin' = Linear Regression
            '-lri' = Linear Ridge Regression
            '-pri' = Polynomial Ridge Regression
            '-pol' = Polynomial Regression
            '-sto' = Stochastic Gradient Descent
        - the programme produces only one file: "trained_reg.pkl"