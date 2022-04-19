package dv.cosine.java;

public class Item {
    public int[] elementIds;
    public int itemId;
    public String split;
    public int label;
    public Item(int[] elementIds, int itemId, String split, int label){
        this.elementIds = elementIds;
        this.itemId = itemId;
        this.split = split;
        this.label = label;
    }
}

