package cs446.homework2;

import java.io.File;
import java.io.FileReader;
import java.io.BufferedReader;
import weka.classifiers.Evaluation;
import weka.core.Instances;
import cs446.weka.classifiers.trees.Id3;
import java.io.FileWriter;
import java.io.BufferedWriter;
public class WekaTesterTree {

    public static void main(String[] args) throws Exception {

	if (args.length != 5) {
	    System.err.println("Usage: WekaTesterSGD train-arff-file test-arrf-file test-blind-file");
	    System.exit(-1);
	}

	// Load the data
	Instances data = new Instances(new FileReader(new File(args[0])));
	Instances test_data = new Instances(new FileReader(new File(args[1])));
	
	// The last attribute is the class label
	data.setClassIndex(data.numAttributes() - 1);
	test_data.setClassIndex(test_data.numAttributes() - 1);
	
	// Train on 80% of the data and test on 20%
	
	int folds = 5;
	double sum_accuracy = 0;
	double total_accuracy = 0;
	for(int i=0; i<folds; i++) {

		Id3 classifier_id3 = new Id3();
		classifier_id3.setMaxDepth(Integer.parseInt(args[4]));
		Instances train = data.trainCV(5,i);
		Instances test = data.testCV(5, i);

		// Train
		classifier_id3.buildClassifier(train);

		// Print the classfier
		//System.out.println(classifier_id3);
		//System.out.println();

		// Evaluate on the test set
		Evaluation evaluation = new Evaluation(test);
		evaluation.evaluateModel(classifier_id3, test);
		System.out.println(evaluation.toSummaryString());

		sum_accuracy = sum_accuracy + evaluation.pctCorrect();
	}

	//This is the total accuracy of the classifier after the 5-folds cross-validation	
	total_accuracy = sum_accuracy/folds;
	System.out.println("Net Accuracy of the classifier: " + total_accuracy);

	//Train the classifier with the entire training set. args[4] gives the max_depth either = -1, 4 or 8
	Id3 classifier_full_test = new Id3();
	classifier_full_test.setMaxDepth(Integer.parseInt(args[4]));
	classifier_full_test.buildClassifier(data);

	//Classify the test data and generate a .pred file
	Instances test_predictions = test_data;
	int numTestInst = test_predictions.numInstances();
	BufferedReader br = new BufferedReader(new FileReader(args[2]));
	BufferedWriter bw = new BufferedWriter(new FileWriter(args[3], true));
	for(int j=0; j<numTestInst; j++){
		
		String currentLine = br.readLine();
		
		if(classifier_full_test.classifyInstance(test_predictions.instance(j)) == 0.0) {
			test_predictions.instance(j).setClassValue("-1");
			currentLine = currentLine.replace("?", "-1");
		} else {
			test_predictions.instance(j).setClassValue("1");
			currentLine = currentLine.replace("?", "1");
		}
		bw.write(currentLine);
		bw.newLine();
		bw.flush();
	}
    }
}
