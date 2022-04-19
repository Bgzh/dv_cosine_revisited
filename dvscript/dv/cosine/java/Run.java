package dv.cosine.java;

import com.google.gson.Gson;
import java.io.FileReader;
import java.io.File;

public class Run {
    public static void main(String[] args){
        Gson gson = new Gson();
        Config config = null;
        try {
            FileReader fr = new FileReader(new File("config.json"));
            config = gson.fromJson(fr, Config.class);
            fr.close();
        } catch(Exception e){
            e.printStackTrace();
        }
        Dataset dataset = new Dataset(config);
        DBOWEmbedding model = new DBOWEmbedding(config);
        model.train(dataset);
    }
    
}
