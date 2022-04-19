package dv.cosine.java;

import java.io.BufferedReader;
import java.io.File;
import java.io.FileReader;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import static java.lang.Math.round;
import static java.lang.Math.pow;
import static java.lang.Math.log;
import static java.lang.Math.abs;
import static java.lang.Math.exp;
import static java.lang.Math.min;
import java.util.Random;
import com.google.gson.Gson;

public class Dataset {
    private double power;
    public int[] elementIdCounts;
    private int[] elementIdSums;
    public List<Item> allItems;
    public int nDim;
    public boolean nb;
    public double[] nbWeights;

    private Random random;

    public Dataset(){
        allItems = new ArrayList<Item>();
    }

    public Dataset(Config config){
        nb = config.nb;
        allItems = new ArrayList<Item>();
        loadDataset(config.filename, config.nDim);
    }

    public void loadDataset(String filename, int nDim) {
        random = new Random(121212);
        power = 0.75;
        this.nDim = nDim;

        try (BufferedReader br = new BufferedReader(new FileReader(new File(filename)))){
            String line;
            Gson gson = new Gson();
            Item item;
            while ((line = br.readLine()) != null) {
                item = gson.fromJson(line, Item.class);
                allItems.add(item);
            }
        } catch (Exception e){
            e.printStackTrace();
        }
        initSum();
    }

    private void initSum(){
        elementIdCounts = new int[nDim];
        elementIdSums = new int[nDim];
        for (Item item: allItems){
            for (int id: item.elementIds){
                elementIdCounts[id] ++;
            }
        }
        elementIdSums[0] = (int)round(pow(elementIdCounts[0], power));
        for (int i=1; i<nDim; i++)
            elementIdSums[i] = (int)round(pow(elementIdCounts[i], power)) + elementIdSums[i-1];
        nbWeights = new double[nDim];
        if (nb) {
            int[] posCounts = new int[nDim];
            int[] negCounts = new int[nDim];
            long posCount = 0;
            long negCount = 0;
            Arrays.fill(posCounts, 1);
            Arrays.fill(negCounts, 1);
            for (Item item: allItems){
                if (item.split.equals("train")){
                    boolean isPos = false;
                    if (item.label == 1)
                        isPos = true;
                    for (int id: item.elementIds){
                        if (isPos){
                            posCounts[id] += 1;
                            posCount++;
                        } else {
                            negCounts[id] += 1;
                            negCount++;
                        }
                    }
                }
            }
            double logPos = log(posCount + nDim);
            double logNeg = log(negCount + nDim);
            double logPN = logPos - logNeg;
            for (int i=0; i<nDim; i++){
                nbWeights[i] = abs(log(posCounts[i]*1.) - log(negCounts[i]*1.) -logPN);
            }
            double mm = 0, cc = 0;
            for (int i=0; i<nDim; i++){
                mm += min(1., exp(nbWeights[i]/0.2)/100) * (posCounts[i] + negCounts[i] - 2);
                cc += posCounts[i] + negCounts[i] - 2;
            } 
            System.out.println(posCount);
            System.out.println(negCount);
            System.out.println(mm/cc);
        }
    }

    public int getRandomElementId(){
        int i = random.nextInt(elementIdSums[nDim-1]) + 1;
        int l = 0, r = nDim - 1;
        while (l != r) {
            int m = (l + r) / 2;
            if (i <= elementIdSums[m]) {
                r = m;
            } else {
                l = m + 1;
            }
        }
        return l;

    }
}
