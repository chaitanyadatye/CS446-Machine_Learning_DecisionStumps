package cs446.homework2;

import weka.core.FastVector;
import java.io.File;
import java.io.FileReader;
import java.io.BufferedReader;
import weka.classifiers.Evaluation;
import weka.core.Instances;
import cs446.weka.classifiers.trees.Id3;
import java.io.FileWriter;
import java.io.BufferedWriter;
import java.util.Random;
import cs446.homework2.DecisionStumps;

public class WekaTesterDecisionStumps {

    public static void main(String[] args) throws Exception {
	
	if (args.length != 3) {
	    System.err.println("Usage: WekaTesterSGD train-arff-file test-arrf-file test-blind-file");
	    System.exit(-1);
	}

	// Load the data
	Instances data = new Instances(new FileReader(new File(args[0])));
	Instances test_data = new Instances(new FileReader(new File(args[1])));

	// The last attribute is the class label
	data.setClassIndex(data.numAttributes() - 1);
	test_data.setClassIndex(test_data.numAttributes() - 1);
	
	int folds = 5;
	double sum_accuracy = 0;
	double total_accuracy = 0;

	//Generate decision stumps for the training dataset
	DecisionStumps decisionStumps = new DecisionStumps();
	decisionStumps.buildClassifier(data);

	//Get new feature vector 2000x100 for the training dataset	
	Instances new_features = decisionStumps.makeDataSet(data);
	new_features.setClassIndex(new_features.numAttributes()-1);
	
	//5-folds cross validation on the SGD of the new training dataset
	for(int i=0; i<folds; i++){	
		
		Instances train = new_features.trainCV(5,i);
		Instances test = new_features.testCV(5,i);

		SGD classifier_sgd = new SGD();
		classifier_sgd.buildClassifier(train);
	
		System.out.println(classifier_sgd);
        	System.out.println();

       		// Evaluate on the test set
        	Evaluation evaluation = new Evaluation(test);
        	evaluation.evaluateModel(classifier_sgd, test);
        	System.out.println(evaluation.toSummaryString());
		
		sum_accuracy = sum_accuracy + evaluation.pctCorrect();
	}
	
	//This is total accuracy of the SGD after the 5-folds cross-validation
	total_accuracy = sum_accuracy/folds;
	System.out.println("Net Accuracy is : " + total_accuracy);

	//Train the classifier on the entire training data
	SGD classifier_sgd_test = new SGD();
	classifier_sgd_test.buildClassifier(new_features);

	//Build the test data set and get the test feature vector 2000x100
	Instances test_predictions = test_data;
	Instances new_test_features = decisionStumps.makeDataSet(test_predictions);

	//Classify the test data and generate a .pred file
	int numTestInst = new_test_features.numInstances();
	BufferedReader br = new BufferedReader(new FileReader(args[2]));
	BufferedWriter bw = new BufferedWriter(new FileWriter("2.e.pred", true));
	for(int j=0; j<numTestInst; j++){
		
		String currentLine = br.readLine();
		
		if(classifier_sgd_test.classifyInstance(new_test_features.instance(j)) == 0.0) {
			new_test_features.instance(j).setClassValue("-1");
			currentLine = currentLine.replace("?", "-1");
		} else {
			new_test_features.instance(j).setClassValue("1");
			currentLine = currentLine.replace("?", "1");
		}
		bw.write(currentLine);
		bw.newLine();
		bw.flush();
	}
    }
}
