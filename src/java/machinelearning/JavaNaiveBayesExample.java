/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package machinelearning;

/**
 *
 * @author PCH
 */
// $example on$
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.function.ToDoubleFunction;
import org.apache.commons.lang.ArrayUtils;
import scala.Tuple2;
import org.apache.spark.api.java.JavaRDD;
import org.apache.spark.api.java.JavaSparkContext;
import org.apache.spark.mllib.classification.NaiveBayes;
import org.apache.spark.mllib.classification.NaiveBayesModel;
import org.apache.spark.mllib.regression.LabeledPoint;
// $example off$
import org.apache.spark.SparkConf;
import org.apache.spark.api.java.function.Function;
import org.apache.spark.api.java.function.PairFunction;
import org.apache.spark.mllib.linalg.Vector;
import org.apache.spark.mllib.linalg.Vectors;


public class JavaNaiveBayesExample {
    
  public static void main(String[] args) {
      
    SparkConf sparkConf = new SparkConf().setAppName("JavaNaiveBayesExample").setMaster("local");
    JavaSparkContext jsc = new JavaSparkContext(sparkConf);

    // $example on$
    String path = "src/resources/sample_libsvm_data.txt";
//    JavaRDD<LabeledPoint> inputData = MLUtils.loadLibSVMFile(jsc.sc(), path).toJavaRDD();

//To read the file
      JavaRDD<String> train_csv = jsc.textFile("src/resources/train_master.csv");
      JavaRDD<String> test_csv = jsc.textFile("src/resources/test_master.csv");
      
JavaRDD<LabeledPoint> training = train_csv.map(new ParsePoint());
JavaRDD<LabeledPoint> test = test_csv.map(new ParsePoint());
//      List<LabeledPoint> mylist = mypoint.collect();
//for(LabeledPoint p: mylist){
//    
//    System.out.println(p.toString());
//    
//}

//    JavaRDD<LabeledPoint> inputData = 
//             MLUtils.loadLibSVMFile(jsc.sc(), path).toJavaRDD();


//    JavaRDD<LabeledPoint>[] tmp = inputData.randomSplit(new double[]{0.6, 0.4});
//    JavaRDD<LabeledPoint> training = tmp[0]; // training set
//    JavaRDD<LabeledPoint> test = tmp[1]; // test set
    NaiveBayesModel model = NaiveBayes.train(training.rdd(), 1.0);
    
//    JavaPairRDD<Double, Double> predictionAndLabel =
//      test.mapToPair(p -> new Tuple2<>(model.predict(p.features()), p.label()));
    
    JavaRDD<Tuple2<LabeledPoint,Double>> predictionAndLabel =
      test.map(new ParsePoint22(model)); 
    
    JavaRDD<LabeledPoint> newRDD = predictionAndLabel.map(new ParsePoint4());
    
//    List<LabeledPoint> mylist =  newRDD.collect();
//    for(LabeledPoint p:mylist){
//        
//        System.out.println(p.toString());
//                
//    }
//    double accuracy = predictionAndLabel.filter(pl -> pl._1().equals(pl._2()) ).count() / (double) test.count();

    double accuracy = 
            newRDD.filter(new ParsePoint3()).count() / (double) test.count();

    // Save and load model
    model.save(jsc.sc(), "target/tmp/myNaiveBayesModel");
    NaiveBayesModel sameModel = NaiveBayesModel.load(jsc.sc(), "target/tmp/myNaiveBayesModel");

    // $example off$

    jsc.stop();
  }
//  public <K2,V2> JavaPairRDD<K2,V2> mapToPair(PairFunction<T,K2,V2> f)
//  filter(Function<Tuple2<K,V>,Boolean> f)
static class ParsePoint3 implements Function<LabeledPoint,Boolean> {  
//pl -> ( pl._1().label() == pl._2()
        @Override
        public Boolean call(LabeledPoint l) throws Exception {
            boolean res = false;
            Vector v = l.features();
            double[] d = v.toArray();
            
            if (l.label() == d[0]){
                res = true;
            }

            return  res;
        }
    
}
  
static class ParsePoint2 implements PairFunction<LabeledPoint,LabeledPoint,Double> {
    NaiveBayesModel model;
    
public ParsePoint2(NaiveBayesModel model){
    this.model=model;
}
        @Override
        public Tuple2<LabeledPoint, Double> call(LabeledPoint t) throws Exception {
          
          double prediction = model.predict(t.features());
          return new Tuple2(t.features(),prediction);
        }

}  
  
static class ParsePoint22 implements Function<LabeledPoint,Tuple2<LabeledPoint,Double>> {
    NaiveBayesModel model;
    
public ParsePoint22(NaiveBayesModel model){
    this.model=model;
}
        @Override
        public Tuple2<LabeledPoint, Double> call(LabeledPoint t) throws Exception {
    
          double prediction = model.predict(t.features());

          return new Tuple2(t,prediction);
        }

}  

//  public <R> JavaRDD<R> map(Function<T,R> f)
static class ParsePoint implements Function<String, LabeledPoint> {

    @Override
    public LabeledPoint call(String line) {
//        System.out.println(line);
        String[] array = line.split(",");
        for(int i=0;i<array.length;i++){
           if(array[i].isEmpty()){
               array[i]="0.0";
           }
        }
        String array_label = Arrays.copyOfRange(array, 0, 1)[0];
        String[] array_vector = Arrays.copyOfRange(array, 1, array.length);

        double[] doubleValues = Arrays.stream(array_vector)
                                  .mapToDouble(new ToDoubleFunction(){
            @Override
            public double applyAsDouble(Object value) {
                return Double.parseDouble((String) value);
            }
        })
                
//                                .mapToDouble(Double::parseDouble)
                                .toArray();
        double label = Double.valueOf(array_label);
        Vector vector = Vectors.dense(doubleValues);
      return new LabeledPoint(label,vector );
    }
  }
  
        
static class ParsePoint4 implements Function<Tuple2<LabeledPoint,Double>, LabeledPoint> {

    @Override
    public LabeledPoint call(Tuple2<LabeledPoint,Double> t) {

        LabeledPoint l = t._1;
        double predict = t._2;
        
        double label = l.label();
        double[] v = l.features().toArray();

        List<Double> mylist = new ArrayList<>();
        mylist.add(label);
        
        for(double d: v){
        mylist.add(d);
        }
            
        Double[] newv = new Double[mylist.size()];
        newv = mylist.toArray(newv);
                        
        double[] d = ArrayUtils.toPrimitive(newv);
            
        Vector dv = Vectors.dense(d);
                      
      return new LabeledPoint(predict, dv);
    }
  }
  
//static class ParsePoint4 implements Function<Tuple2<LabeledPoint,Double>, LabeledPoint> {
//
//    @Override
//    public LabeledPoint call(Tuple2<LabeledPoint,Double> t) {
//
//        LabeledPoint l = t._1;
//        double predict = t._2;
//        double[] init = {0.0,0.0,0.0,0.0,0.0};
//        org.apache.spark.mllib.linalg.DenseVector dense = new DenseVector(init);
//        Vector dv = Vectors.dense(dense.values());
//        Object o = t._1;
//        
//        if (o instanceof DenseVector){
//            dense = (org.apache.spark.mllib.linalg.DenseVector)o;
//            double[] v = dense.toArray();
//
//            List<Double> mylist = new ArrayList<>();
//            mylist.add(predict);
//            for(double d: v){
//                mylist.add(d);
//            }
//            
//            Double[] newv = new Double[mylist.size()];
//            newv = mylist.toArray(newv);
//                        
//            double[] d = ArrayUtils.toPrimitive(newv);
//            
//            dv = Vectors.dense(d);
//        }
//              
//      return new LabeledPoint(predict, dv);
//    }
//  }

}
