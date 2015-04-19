package cs446.homework2;

import weka.classifiers.*;
import weka.core.Instances;
import weka.core.Instance;
import java.lang.*;
import java.util.*;
import java.util.Arrays;
import java.lang.Math;

public class SGD extends Classifier{

	private boolean trained = false;
	private double weight_0 = 0.0;		//theta or w_0
	private double Rate = 0.001;		//learning rate
	private int numAttr;			
	private int numInst;
	private double weight[];		//weight vector

	
/* 
 * This function takes in a class label index of type double and returns the actual label corresponding to that index
 */ 
	public double sign(double val) {
		if (val == 0.0)
			return -1.0;
		else
			return 1.0;
	}
/*
 * Initialize the class members
 */
	public void initialize(Instances instances) {
	
		numInst = instances.numInstances();
		numAttr = instances.numAttributes() - 1;
		weight = new double[numAttr];
		Arrays.fill(weight, 0.0);
		/*Random random = new Random();
		for(int i=0; i<numAttr; i++) {
			weight[i] = random.nextDouble();
		}*/

	}
/*
 * This function predicts the value of a particular instance
 * It calculates sum of all w*x products and adds w_0 to it
 * returns the predicted value for that instance. If w*x+w_0 >= 0 -> return 1 else return -1
 */
	public double predicted_value(Instance instofi) {
		double pred = 0.0;
		for(int i=0; i<numAttr; i++) {
			double attrValue = Double.parseDouble(instofi.stringValue(i));
			pred = pred + weight[i]*attrValue;
		}
		pred = pred + weight_0;
		if(pred >=0 )
			return 1.0;
		else
			return -1.0;
	}
	
/*
 * This function returns the actual value of the class label
 */
	public double actual_value(Instance instofi) {
		return sign(instofi.classValue());
	}
	
/*
 * Calculates the error corresponding to the current w 
 */
	public double calculate_error(Instances instances) {
		double error = 0.0;
		for(int i=0; i<numInst; i++) {
			Instance instofi = instances.instance(i);
			error = error + Math.pow((predicted_value(instofi) - actual_value(instofi)), 2);
		}
		return error/2;
	}
	
	@Override
	public void buildClassifier(Instances arg0) throws Exception {
		
		initialize(arg0);	
		if(arg0.numInstances()==0){
			System.out.println("ZERO INSTANCES");
		}
		//lets train it using 1000 epochs
		for(int n=1000; n>0; n--){
		for(int i=0; i<numInst; i++){
			Instance instofi = arg0.instance(i);
			double predictedVal = predicted_value(instofi);
			double actualVal = actual_value(instofi);
			//error = calculate_error(arg0);
			//System.out.println("Error : " + error);
			//System.out.println("Predicted : Actual " + predictedVal + " " + actualVal);
			for (int j=0; j<numAttr; j++) {
				double attrValue = Double.parseDouble(instofi.stringValue(j));
						
				weight[j] = weight[j] + Rate*(actualVal - predictedVal)*attrValue;
		}
		weight_0 = weight_0 + Rate*(actualVal - predictedVal);
		}
		
		}
		trained = true;
	}

	@Override
	public double classifyInstance(Instance instance) throws java.lang.Exception {
		if(!trained){
			throw new Exception("The classifier is not trained!");
		}
		double predictedVal = predicted_value(instance);
		
		if(predictedVal == 1.0)
			return 1.0;
		else
			return 0.0;		
		
		}
	
}


