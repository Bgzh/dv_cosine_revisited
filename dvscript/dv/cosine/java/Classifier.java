package dv.cosine.java;

import de.bwaldvogel.liblinear.FeatureNode;
import de.bwaldvogel.liblinear.Linear;
import de.bwaldvogel.liblinear.Model;
import de.bwaldvogel.liblinear.Parameter;
import de.bwaldvogel.liblinear.Problem;
import de.bwaldvogel.liblinear.SolverType;
import java.util.List;

public class Classifier {
    
    private Parameter parameter;
    private Model model;
    
    public Classifier(SolverType solver, double C, double eps) {
        parameter = new Parameter(solver, C, eps);
    }
    
    public void train(double[][] WP, List<Item> trainSet) {
        Problem problem = new Problem();
        int numInstances = trainSet.size();
        int numFeatures = WP[0].length;
        problem.l = numInstances;
        problem.n = numFeatures;
        
        FeatureNode[][] X_train = new FeatureNode[numInstances][numFeatures];
        double[] Y_train = new double[numInstances];
        for (int i=0; i<numInstances; i++){
            Item doc = trainSet.get(i);
            Y_train[i] = doc.label;
            for (int j=0; j<numFeatures; j++) {
                X_train[i][j] = new FeatureNode(j+1,WP[doc.itemId][j]);
            }
        }
        problem.x = X_train;
        problem.y = Y_train;
        Linear.setDebugOutput(null);
        model = Linear.train(problem, parameter);
    }
    
    public double score(double[][] WP, List<Item> testSet) {
        int numInstances = testSet.size();
        int numFeatures = WP[0].length;
        
        int corrects = 0;
        FeatureNode[] X_test = new FeatureNode[numFeatures];
        for (int i=0; i<numInstances; i++){
            Item doc = testSet.get(i);
            double Y_test = doc.label;
            for (int j=0; j<numFeatures; j++) {
                X_test[j] = new FeatureNode(j+1,WP[doc.itemId][j]);
            }
            double prediction = Linear.predict(model, X_test);
            if (Y_test == prediction) {
                corrects++;
            }
        }
        
        double accuracy = ((corrects+0.0)/numInstances)*100;
        // System.out.println("Accuracy = "+ accuracy +"% ("+ corrects+"/"+numInstances+")");
        return accuracy;
    }

    public double crossValidate(double[][] WP, List<Item> trainSet) {
        Problem problem = new Problem();
        int numInstances = trainSet.size();
        int numFeatures = WP[0].length;
        problem.l = numInstances;
        problem.n = numFeatures;
        double[] targets = new double[numInstances];
        double acc = 0.;
        
        FeatureNode[][] X_train = new FeatureNode[numInstances][numFeatures];
        double[] Y_train = new double[numInstances];
        for (int i=0; i<numInstances; i++){
            Item doc = trainSet.get(i);
            Y_train[i] = doc.label;
            for (int j=0; j<numFeatures; j++) {
                X_train[i][j] = new FeatureNode(j+1,WP[doc.itemId][j]);
            }
        }
        problem.x = X_train;
        problem.y = Y_train;
        Linear.setDebugOutput(null);
        Linear.crossValidation(problem, parameter, 5, targets);
        int corrects = 0;
        for (int i=0; i<numInstances; i++){
            if (targets[i] == Y_train[i])
                corrects ++;
        }
        acc = (corrects + 0.0) / numInstances * 100;
        //System.out.println("CvAccuracy = "+ acc +"% ("+ corrects+"/"+numInstances+")");
        return acc;
    }
}
