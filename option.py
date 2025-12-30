class Options:
    def __init__(self):
        self.datasetname = "LK"
        self.numtrain = 0.01
        self.batchSize = 32
        self.epochs = 70
        self.spectrumnum = 36
        self.inputsize = 7
        self.windowsize = 3
        self.sampling_mode = "random"
        self.input3D = False
        self.nz = 100
        self.D_lr = 0.001
        self.cuda = True
        self.netD = ''
        self.manualSeed = 531
        self.random_seed = 5

opt = Options()
output_units = 16
