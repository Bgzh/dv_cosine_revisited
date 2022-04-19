package dv.cosine.java;

import de.bwaldvogel.liblinear.SolverType;
import java.io.FileWriter;
import java.io.IOException;
import static java.lang.Math.exp;
import static java.lang.Math.pow;
import static java.lang.Math.sqrt;
import static java.lang.Math.log;
import static java.util.stream.Collectors.toList;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.List;
import java.util.Random;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.TimeUnit;
import com.google.gson.Gson;

public class DBOWEmbedding {
    private int n; // number of dimensions of the embeddings
    private int minTf;
    private double lr;
    private boolean subSamp;
    private int nEpoch;

    private double nbA; // parameters for subSamp
    private double nbB; // the prob to be sampled is exp(NBW / nbA) / nbB

    private long randomSeed;
    private int earlyStoppingPatience; // 0: no early stopping

    private int negSize=5;
    private double a = 6; // for cosine
    private int batchSize = 100; // number of items in each thread;
    private int numThreads = 22;

    private String vecPath;
    private String logPath;

    private Dataset dataset;
    private double [][] wordVecs;  //embeddings for elements / ngrams /words
    private double [][] docVecs;  //embeddings for items / docs

    private boolean test;
    private double [] Cs;  // grid of C for logstic regression
    private int verbose;  // 0: silent, 1: almost silent, 2: not silent
    
    private double dvNormSum;
    private double gradNormSum;
    private double lossSum;
    private long wordCountSum;
    private int m = 0; // number of items

    private Random random;

    public DBOWEmbedding (Config config) {
        this.n = config.n;
        this.minTf = config.minTf;
        this.lr = config.lr;
        this.nEpoch = config.nEpoch;

        this.subSamp = config.subSamp;
        this.nbA = config.nbA;
        this.nbB = config.nbB;

        this.randomSeed = config.randomSeed;
        this.earlyStoppingPatience = config.earlyStoppingPatience;

        this.vecPath = config.vecPath;
        this.logPath = config.logPath;
        this.test = config.test;
        this.Cs = config.Cs;
        this.verbose = config.verbose;
        random = new Random(this.randomSeed);
        dataset = null;
    }

    public void train(Dataset dataset) {
        this.dataset = dataset;
        List<Item> docList = new ArrayList<Item>(dataset.allItems);
        List<Item> trainSet = new ArrayList<Item>();
        List<Item> devSet = new ArrayList<Item>();
        List<Item> testSet = new ArrayList<Item>();
        for (Item item : docList) {
            if (item.split.equals("train"))
                trainSet.add(item);
            else if (item.split.equals("dev"))
                devSet.add(item);
            else if (item.split.equals("test"))
                testSet.add(item);
        }
        if (verbose>1)
            System.out.printf("train %d dev %d test %d\n", trainSet.size(), devSet.size(), testSet.size());
        m = dataset.allItems.size();

        double bestDevAccEver = 0.;
        int bestEpoch = 0;
        
        double accuracy = 0.;
        double bestDevAcc = 0.;
        initEmbs();

        for (int epoch=0; epoch<nEpoch; epoch++){
            int startEpoch = (int) System.currentTimeMillis();
            if (verbose>1)
                System.out.printf("Epoch %d:\n", epoch);

            dvNormSum = 0.;
            gradNormSum = 0.;
            lossSum = 0.;
            wordCountSum = 0;

            int p=0;
            Collections.shuffle(docList);
            ExecutorService pool = Executors.newFixedThreadPool(numThreads);
            while (p < m / batchSize){
                int s = batchSize * p;
                int e = batchSize * p + batchSize;
                if (m < e)
                    e = m;
                pool.execute(new TrainThread(docList.subList(s, e)));
                p += 1;
            }
            pool.shutdown();
            try {
                pool.awaitTermination(Long.MAX_VALUE, TimeUnit.NANOSECONDS);
            } catch (InterruptedException e) {
                e.printStackTrace();
            }
            int endEpoch = (int) System.currentTimeMillis();
            if (verbose>1)
                System.out.printf("time: %d seconds\n", (endEpoch - startEpoch) / 1000);

            bestDevAcc = 0.;
            double bestC = 0.;
            double acc;
            if (test){
                for (double C: Cs){
                    Classifier bClf = new Classifier(SolverType.L2R_LR, C, 0.01);
                    bClf.train(docVecs, trainSet);
                    acc = bClf.score(docVecs, devSet);
                    if (acc > bestDevAcc){
                        bestDevAcc = acc;
                        bestC = C;
                    }
                }
                if (verbose>1){
                    System.out.printf("dev acc %.2f\n", bestDevAcc);
                }
                Classifier bClf = new Classifier(SolverType.L2R_LR, bestC, 0.01);
                bClf.train(docVecs, trainSet);
                accuracy = bClf.score(docVecs, testSet);
            }

            double dvNorm;
            for (int i=0; i<m; i++){
                dvNorm = 0;
                for (int j=0; j<n; j++){
                    dvNorm += pow(docVecs[i][j], 2);
                }
                dvNormSum += pow(dvNorm, 0.5);
            }

            try {
                FileWriter fw = new FileWriter(logPath, true);
                fw.write(String.format("epoch %d mean_dv_norm %f mean_grad_norm %f mean_ngram_count %d mean_loss %f C %f dev_acc %f test_acc %f\n", 
                    epoch, dvNormSum / m, gradNormSum / wordCountSum, wordCountSum / m, lossSum / wordCountSum,
                    bestC, bestDevAcc, accuracy));
                fw.close();
            } catch (IOException e) {
                e.printStackTrace();
            }

            if (earlyStoppingPatience > 0) {
                if (bestDevAcc > bestDevAccEver){
                    bestDevAccEver = bestDevAcc;
                    bestEpoch = epoch;
                    saveVecs();
                }
                if (epoch > bestEpoch + earlyStoppingPatience)
                    break;
            }
        }
        if (earlyStoppingPatience == 0)
            saveVecs();
        if (verbose==1)
            System.out.printf("final dev acc %.2f\n", bestDevAcc);
        if (verbose>0)
            System.out.println("done");
    }

    private void initEmbs(){
        wordVecs = new double[dataset.nDim][];
        for (int i=0; i<dataset.nDim; i++){
            if (dataset.elementIdCounts[i] >= minTf){
                wordVecs[i] = new double[n];
                for (int j=0; j<n; j++){
                    wordVecs[i][j] = (random.nextFloat() - 0.5) / n;
                }
            }
        }
        docVecs = new double[m][n];
        for (int i=0; i<m; i++){
            for (int j=0; j<n; j++)
                docVecs[i][j] = (random.nextFloat() - 0.5) / n;
        }
    }

    private class TrainThread implements Runnable {
        private List<Item> subList;
        private long wordCountSumThread;
        private double lossSumThread;
        private double gradNormSumThread;
        public TrainThread(List<Item> subList){
            this.subList = subList;
            wordCountSumThread = 0;
            lossSumThread = 0;
            gradNormSumThread = 0;
        }
        public void run() {
            train();
        }
        private void train(){
            double [] temp = new double[n];
            for (Item item: subList){
                int ii = item.itemId;
                List<Integer> ids = Arrays.stream(item.elementIds).boxed().collect(toList());
                Collections.shuffle(ids, random);

                double gradNorm;
                for (int i: ids){
                    if (subSamp){
                        if (random.nextDouble() > exp(dataset.nbWeights[i] / nbA) / nbB)
                            continue;
                    }
                    Arrays.fill(temp, 0);
                    backprop(docVecs[ii], wordVecs[i], 1., temp);
                    for (int ni=0; ni<negSize; ni++){
                        backprop(docVecs[ii], wordVecs[dataset.getRandomElementId()], 0, temp);
                    }
                    for (int j=0; j<n; j++){
                        docVecs[ii][j] += temp[j];
                    }
                    
                    wordCountSumThread += 1;
                    gradNorm = 0.;
                    for (int j=0; j<n; j++){
                        gradNorm += pow(temp[j], 2);
                    }
                    gradNormSumThread += pow(gradNorm, 0.5) / lr;
                }
            }
            wordCountSum += wordCountSumThread;
            gradNormSum += gradNormSumThread;
            lossSum += lossSumThread;
        }

        private void backprop(double[] dv, double[] wv, double t, double[] temp){
            if (wv == null)
                return;
            double dot = 0;
            double normDv = 0;
            double normWv = 0;
            for (int i=0; i<n; i++){
                dot += dv[i] * wv[i];
                normDv += dv[i] * dv[i];
                normWv += wv[i] * wv[i];
            }
            normDv = sqrt(normDv);
            normWv = sqrt(normWv);
            double cos_theta = dot / (normDv * normWv);
            double y = 1.0 / (1. + exp(-a * cos_theta));
            if (t==1)
                lossSumThread += -log(y);
            else if (t==0)
                lossSumThread += -log(1-y);
            double A = -(y - t) * a / (normDv * normWv) * lr;
            double E = -(y - t) * a * cos_theta * lr;
            double B = E / pow(normDv, 2);
            double C = E / pow(normWv, 2);
            for (int i=0; i<n; i++){
                temp[i] += wv[i] * A - dv[i] * B;
                wv[i] += dv[i] * A - wv[i] * C;
            }        
        }
    }

    private void saveVecs(){
        Gson gson = new Gson();
        Vec vec = null;
        try{
            FileWriter fw = new FileWriter(vecPath);
            for (Item item: dataset.allItems){
                if (item.split.equals("extra"))
                    continue;
                int id = item.itemId;
                vec = new Vec(docVecs[id], id, item.split, item.label);
                fw.write(gson.toJson(vec));
                fw.write('\n');
            }
            fw.close();
        } catch (IOException e){
            e.printStackTrace();
        }
    }




    
}
