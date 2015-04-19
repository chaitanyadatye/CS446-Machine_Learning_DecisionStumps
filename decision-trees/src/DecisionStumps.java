package cs446.homework2;

import weka.core.Attribute;
import weka.classifiers.*;
import java.io.File;
import java.io.FileReader;
import java.io.BufferedReader;
import weka.classifiers.Evaluation;
import weka.core.Instances;
import cs446.weka.classifiers.trees.Id3;
import java.io.FileWriter;
import java.io.BufferedWriter;
import java.util.Random;
import weka.core.Instance;
import weka.core.FastVector;
public class DecisionStumps extends Classifier{

	private boolean trained = false;
	private Id3 classifier_stumps[];
	Random randnum;

	public DecisionStumps() {
		classifier_stumps = new Id3[100];
		randnum = new Random();
		randnum.setSeed(123456789);
	}
	
	@Override
	public void buildClassifier(Instances data) throws Exception {
		
		for(int i=0; i<100; i++){
		
			//Randomize the training dataset	
			data.randomize(randnum);
			int trainsize = (int) Math.round(data.numInstances() *0.5);
			
			//new_data is the new training data which is 50% of the original data and in a randomized fashion
			Instances new_data = new Instances(data, 0, trainsize);
				
			classifier_stumps[i] = new Id3(); 
			classifier_stumps[i].setMaxDepth(4);
			classifier_stumps[i].buildClassifier(new_data);
	
			//System.out.println(classifier_stumps[i]);
                	//System.out.println();

                	//Evaluation evaluation = new Evaluation(new_data);
                	//evaluation.evaluateModel(classifier_stumps[i], new_data);
                	//System.out.println(evaluation.toSummaryString());
		}
	trained = true;	
	}

/*
 * Create a new feature set from the original dataset
 * This function will create a new feature set which is of dimension 2000x100
 * This new feature set will be used by the SGD to learn 
 */
	public Instances makeDataSet(Instances data) throws Exception{
		
		FastVector zeroOne = new FastVector(2);
		FastVector labels = new FastVector(2);
		zeroOne.addElement("1");
		zeroOne.addElement("0");

		labels.addElement("-1");
		labels.addElement("1");
			
		FastVector attributes = new FastVector(101);
		String base = "decisionstumps";
		for(int i=0; i<100; i++){
		
			Attribute attr = new Attribute(base+i, zeroOne);
			attributes.addElement(attr);
		}
		Attribute classLabel = new Attribute("Class", labels);
		attributes.addElement(classLabel);

		String newDataSet = "NewDataSet";
		Instances new_data = new Instances(newDataSet, attributes, data.numInstances());
		new_data.setClassIndex(new_data.numAttributes()-1);	
			
		int numInst = data.numInstances();
		for(int i=0; i<numInst; i++) {
			Instance instance = new Instance(new_data.numAttributes());
			instance.setDataset(new_data);
			
			for(int j=0; j<100; j++) {
				double attrVal = classifier_stumps[j].classifyInstance(data.instance(i));
				instance.setValue(j, attrVal);
			}
			String label;
			if(data.instance(i).classValue() == 0.0)
				label = "-1";
			else
				label = "1";
			
			instance.setClassValue(label);
			new_data.add(instance);

		}
	return new_data;
	}

}
