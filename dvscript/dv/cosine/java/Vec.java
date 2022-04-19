package dv.cosine.java;

public class Vec {
    public double[] embs;
    public int itemId;
    public String split;
    public int label;
    public Vec(double[] embs, int itemId, String split, int label){
        this.embs = embs;
        this.itemId = itemId;
        this.split = split;
        this.label = label;
    }

}
