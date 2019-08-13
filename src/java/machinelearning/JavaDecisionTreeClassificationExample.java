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
import java.util.HashMap;
import java.util.Map;

import scala.Tuple2;

import org.apache.spark.SparkConf;
import org.apache.spark.api.java.JavaPairRDD;
import org.apache.spark.api.java.JavaRDD;
import org.apache.spark.api.java.JavaSparkContext;
import org.apache.spark.api.java.function.Function;
import org.apache.spark.api.java.function.PairFunction;
import org.apache.spark.mllib.regression.LabeledPoint;
import org.apache.spark.mllib.tree.DecisionTree;
import org.apache.spark.mllib.tree.model.DecisionTreeModel;
import org.apache.spark.mllib.util.MLUtils;
// $example off$

class JavaDecisionTreeClassificationExample {

  public static void main(String[] args) {

    // $example on$
    SparkConf sparkConf = new SparkConf().setAppName("JavaDecisionTreeClassificationExample");
    JavaSparkContext jsc = new JavaSparkContext(sparkConf);

    // Load and parse the data file.
    String datapath = "data/mllib/sample_libsvm_data.txt";
    JavaRDD<LabeledPoint> data = MLUtils.loadLibSVMFile(jsc.sc(), datapath).toJavaRDD();
    // Split the data into training and test sets (30% held out for testing)
    JavaRDD<LabeledPoint>[] splits = data.randomSplit(new double[]{0.7, 0.3});
    JavaRDD<LabeledPoint> trainingData = splits[0];
    JavaRDD<LabeledPoint> testData = splits[1];

    // Set parameters.
    //  Empty categoricalFeaturesInfo indicates all features are continuous.
    int numClasses = 2;
    Map<Integer, Integer> categoricalFeaturesInfo = new HashMap<>();
    String impurity = "gini";
    int maxDepth = 5;
    int maxBins = 32;

    // Train a DecisionTree model for classification.
    DecisionTreeModel model = DecisionTree.trainClassifier(trainingData, numClasses,
      categoricalFeaturesInfo, impurity, maxDepth, maxBins);

    // Evaluate model on test instances and compute test error
//    JavaPairRDD<Double, Double> predictionAndLabel =
//      testData.mapToPair(p -> new Tuple2<>(model.predict(p.features()), p.label()));
    
      JavaPairRDD<Double, Double> predictionAndLabel =
      testData.mapToPair(new ParsePoint1( model));
                  
//    double testErr =
//      predictionAndLabel.filter(pl -> !pl._1().equals(pl._2())).count() / (double) testData.count();

    double testErr =
      predictionAndLabel.filter(new Function<Tuple2<Double,Double>,Boolean>() {
        @Override
        public Boolean call(Tuple2<Double, Double> t1) throws Exception {
            return !t1._1().equals(t1._2());
        }
    } ).count() / (double) testData.count();
              

    System.out.println("Test Error: " + testErr);
    System.out.println("Learned classification tree model:\n" + model.toDebugString());

    // Save and load model
    model.save(jsc.sc(), "target/tmp/myDecisionTreeClassificationModel");
    DecisionTreeModel sameModel = DecisionTreeModel
      .load(jsc.sc(), "target/tmp/myDecisionTreeClassificationModel");
    // $example off$
  }
  
static class ParsePoint1 implements PairFunction<LabeledPoint,Double,Double> {
    DecisionTreeModel model;
    
public ParsePoint1(DecisionTreeModel model){
    this.model=model;
}
        @Override
        public Tuple2<Double, Double> call(LabeledPoint t) throws Exception {
            return new Tuple2<>(model.predict(t.features()), t.label());
        }

}   
  
static class ParsePoint2 implements Function<Tuple2<Double,Double>,Boolean> {

        @Override
        public Boolean call(Tuple2<Double, Double> t1) throws Exception {
            throw new UnsupportedOperationException("Not supported yet."); //To change body of generated methods, choose Tools | Templates.
        }

}  
  
}
