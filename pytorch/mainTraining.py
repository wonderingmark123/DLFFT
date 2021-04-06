from utils import Training,Testing

def main():
    Training(
    batchSize   = 64,
    numWorkers  = 0,
    LinerPara   = [800,200,9],
    PinMemory   = True,
    epochs      = 200,
    learning_rate = 1e-4,
    decay       = 1e-6,
    ModelSaveFolder = "..\\data\\Models\\NNliner1000",
    DataFolder      = "..\\data\\GeneratorMatlabAbs/")
    Testing(
    batchSize   = 2,
    numWorkers  = 0,
    LinerPara   = [1000,100,9],
    PinMemory   = True,
    Numtesting      = 100,
    learning_rate = 1e-4,
    decay       = 1e-6,
    ModelSaveFolder = "..\\data\\Models\\NNliner1000",
    DataFolder      = "..\\data\\GeneratorMatlabTest/")
if __name__ == '__main__':
    main()