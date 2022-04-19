package dv.cosine.java;

public class Config {
    // Dataset
    public String filename;
    public int nDim;
    public boolean nb;
    // DBOWEmbedding
    public int n;
    public int minTf;
    public double lr;
    public int nEpoch;

    public boolean subSamp;
    public double nbA;
    public double nbB;

    public long randomSeed = 22;
    public int earlyStoppingPatience = 0;

    public String vecPath;
    public String logPath;

    public boolean test;
    public double [] Cs;
    public int verbose;

    public Config(String filename, int nDim, boolean nb, int n, int minTf, double lr, int nEpoch, 
        boolean subSamp, double nbA, double nbB, String vecPath, String logPath,
        boolean test, double[] Cs, int verbose){
            this.filename = filename;
            this.nDim = nDim;
            this.nb = nb;
            this.n = n;
            this.minTf = minTf;
            this.lr = lr;
            this.nEpoch = nEpoch;
            this.subSamp = subSamp;
            this.nbA = nbA;
            this.nbB = nbB;
            this.vecPath = vecPath;
            this.logPath = logPath;
            this.test = test;
            this.Cs = Cs;
            this.verbose = verbose;
        };

    public Config(String filename, int nDim, boolean nb, int n, int minTf, double lr, int nEpoch, 
        boolean subSamp, double nbA, double nbB, long randomSeed, int earlyStoppingPatience,
        String vecPath, String logPath, boolean test, double[] Cs, int verbose){
            this.filename = filename;
            this.nDim = nDim;
            this.nb = nb;
            this.n = n;
            this.minTf = minTf;
            this.lr = lr;
            this.nEpoch = nEpoch;
            this.subSamp = subSamp;
            this.nbA = nbA;
            this.nbB = nbB;
            this.randomSeed = randomSeed;
            this.earlyStoppingPatience = earlyStoppingPatience;
            this.vecPath = vecPath;
            this.logPath = logPath;
            this.test = test;
            this.Cs = Cs;
            this.verbose = verbose;
        };
}
