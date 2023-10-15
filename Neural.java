// Neural Network Java classes
//


import java.awt.*;
import java.applet.Applet;
import java.lang.*;
import java.util.*;

class Neural extends Object 
{

	// Number of input neurons and input vector
	public int numInputNeurons;
	public double inputNeurons[];

	// Number of hiddenNeurons neurons and hiddenNeurons vector
	public int numHiddenNeurons;
	public double hiddenNeurons[];

	// Number of output classification neurons and output vector
	public int numOutputNeurons;
	public double outputNeurons[];

	// Declare weight arrays
	public double inputToHiddenWeights[][];
	public double hiddenToOutputWeights[][];


	protected int NumTraining;
	protected int WeightsFlag;
	protected int SpecialFlag;

	protected double output_errors[];
	protected double hiddenNeurons_errors[];

	protected double inputTraining[];
	protected double outputTraining[];

	// mask of training examples to ignore (true -> ignore):
	// FUTURE USE [GENETIC ALGORITHMS]
	public boolean IgnoreTraining[] = null;

	// mask of Input neurons to ignore:
	// FUTURE USE  [GENETIC ALGORITHMS]
	public boolean IgnoreInput[] = null;

	//FUTURE USE [SAVING TO/RESTORING FROM FILE]
	public NNfile NeuralFile=null;

	// For debug output:
	GUI MyGUI = null;

	// Default Contructor
	Neural() 
	{
		numInputNeurons = 0;
		numHiddenNeurons = 0;
		numOutputNeurons = 0;
	}
	
	// Main Parameterized Contructor
	Neural(int i, int h, int o) 
	{
		System.out.println("In BackProp constructor");
		inputNeurons = new double[i];
		hiddenNeurons = new double[h];
		outputNeurons = new double[o];
		inputToHiddenWeights = new double[i][h];
		hiddenToOutputWeights = new double[h][o];
		numInputNeurons = i;
		numHiddenNeurons = h;
		numOutputNeurons = o;
		output_errors = new double[numOutputNeurons];
		hiddenNeurons_errors = new double[numHiddenNeurons];

		// Randomize weights here:
		randomizeWeights();
   }


// Randomize initial weights for network
// Weights range from -0.5 to 0.5
public void randomizeWeights() 
{
	// Randomize Input to Hidden weights
	for (int ii = 0; ii < numInputNeurons; ii++)
	{
		for (int hh = 0; hh < numHiddenNeurons; hh++)
		{
			inputToHiddenWeights[ii][hh] = 0.1f * Math.random() - 0.05;
		}
	}

	// Randomize Hidden to Input weights
	for (int hh = 0; hh < numHiddenNeurons; hh++)
	{
		for (int oo = 0; oo < numOutputNeurons; oo++)
		{
			hiddenToOutputWeights[hh][oo] = 0.1f * Math.random() - 0.05;
		}
	}
}

	// Perform a Forward Pass through the network
	// This is used in training and in classification
	public void ForwardPass() 
	{
		// Reset hidden neurons
		for (int h = 0; h < numHiddenNeurons; h++) 
		{
			hiddenNeurons[h] = 0.0;
		}
		
		// Reset output neurons
		for (int o=0; o<numOutputNeurons; o++)
		{
			outputNeurons[o] = 0.0;
		}

		// Start forward pass
		
		// Compute input to each hidden neuron by
		// totaling the value of all input neurons multiplied by
		// the weight connecting each to the hidden neuron
		for (int i = 0; i < numInputNeurons; i++)
		{
			for (int h=0; h<numHiddenNeurons; h++) 
			{
				hiddenNeurons[h] += inputNeurons[i] * inputToHiddenWeights[i][h];
			}
		}
		
		// Compute input to each output neuron by
		// totaling the value of all hidden neurons multiplied by
		// the weight connecting each to the output neuron modified
		// by the sigmoid threshold function
		for (int h=0; h<numHiddenNeurons; h++) 
		{
			for (int o=0; o<numOutputNeurons; o++) 
			{
				outputNeurons[o] += Sigmoid(hiddenNeurons[h]) * hiddenToOutputWeights[h][o];
			}
		}
		
		// Compute the final output neuron value by
		// modifying it by the sigmoid threshold function
		for (int o=0; o<numOutputNeurons; o++)
		{
			outputNeurons[o] = Sigmoid(outputNeurons[o]);
		}
  }

	public double Train() 
	{
		return Train(inputTraining, outputTraining, NumTraining);
	}

	public double Train(double ins[], double outs[], int num_cases) 
	{
		int in_count=0, out_count=0;
		double error = 0.0;
		for (int example=0; example<num_cases; example++) 
		{
		  if (IgnoreTraining != null)
			 if (IgnoreTraining[example]) continue;  // skip this case
		  // zero out error arrays:
		  for (int h=0; h<numHiddenNeurons; h++)
			 hiddenNeurons_errors[h] = 0.0;
		  for (int o=0; o<numOutputNeurons; o++)
			 output_errors[o] = 0.0;
		  // copy the input values:
		  for (int i=0; i<numInputNeurons; i++) 
		  {
			 inputNeurons[i] = ins[in_count++];
		  }

		  if (IgnoreInput != null) {
			for (int ii=0; ii<numInputNeurons; ii++) 
			{
			   if (IgnoreInput[ii]) {
				 for (int hh=0; hh<numHiddenNeurons; hh++) 
				 {
					inputToHiddenWeights[ii][hh] = 0;
				 }
			   }
			}
		  }

		  // perform a forward pass through the network:

		  ForwardPass();

		  if (MyGUI != null)  MyGUI.repaint();
		  for (int o=0; o<numOutputNeurons; o++)  
		  {
			  output_errors[o] = (outs[out_count++] - outputNeurons[o]) * SigmoidP(outputNeurons[o]);
		  }
		  for (int h=0; h<numHiddenNeurons; h++) 
		  {
			hiddenNeurons_errors[h] = 0.0;
			for (int o=0; o<numOutputNeurons; o++) 
			{
			   hiddenNeurons_errors[h] += output_errors[o]*hiddenToOutputWeights[h][o];
			}
		  }
		  for (int h=0; h<numHiddenNeurons; h++) 
		  {
			 hiddenNeurons_errors[h] = hiddenNeurons_errors[h] * SigmoidP(hiddenNeurons[h]);
		  }
		  // update the hiddenNeurons to output weights:
		  for (int o=0; o<numOutputNeurons; o++) 
		  {
			 for (int h=0; h<numHiddenNeurons; h++) 
			 {
				hiddenToOutputWeights[h][o] +=  0.5 * output_errors[o] * hiddenNeurons[h];
			 }
		  }
		  // update the input to hiddenNeurons weights:
		  for (int h=0; h<numHiddenNeurons; h++) 
		  {
			 for (int i=0; i<numInputNeurons; i++) 
			 {
				 inputToHiddenWeights[i][h] +=  0.5 * hiddenNeurons_errors[h] * inputNeurons[i];
			 }
		  }
		  for (int o=0; o<numOutputNeurons; o++)
			  error += Math.abs(output_errors[o]);
		}
		return error;
	}

	// Sigmoid threshold function
	protected double Sigmoid(double x) 
	{
		return ((1.0/(1.0+Math.exp((-x))))-0.5);
	}

	// Back-propagation learning function
	protected double SigmoidP(double x) 
	{
		double z = Sigmoid(x) + 0.5;
		return (z * (1.0 - z));
	}
	
//FUTURE USE [SAVING TO/RESTORING FROM FILE]
/*	
	Neural(String file_name) 
	{
		NeuralFile = new NNfile(file_name);
		numInputNeurons = NeuralFile.NumInput;
		numHiddenNeurons = NeuralFile.numHiddenNeurons;
		numOutputNeurons = NeuralFile.NumOutput;
		NumTraining= NeuralFile.NumTraining;
		WeightsFlag= NeuralFile.WeightFlag;
		SpecialFlag= NeuralFile.SpecialFlag;

		inputNeurons = new double[numInputNeurons];
		hiddenNeurons = new double[numHiddenNeurons];
		outputNeurons = new double[numOutputNeurons];
		inputToHiddenWeights = new double[numInputNeurons][numHiddenNeurons];
		hiddenToOutputWeights = new double[numHiddenNeurons][numOutputNeurons];
		// Retrieve the weight values from the NNfile object:
		if (WeightsFlag!=0) 
		{
			for (int i=0; i<numInputNeurons; i++) 
			{
				for (int h=0; h<numHiddenNeurons; h++) 
				{
					inputToHiddenWeights[i][h] = NeuralFile.GetinputToHiddenWeights(i, h);
				}
			}
			for (int h=0; h<numHiddenNeurons; h++) 
			{
				for (int o=0; o<numOutputNeurons; o++) 
				{
					hiddenToOutputWeights[h][o] = NeuralFile.GethiddenToOutputWeights(h, o);
				}
			}
		} 
		else 
		{
			randomizeWeights();
		}

		output_errors = new double[numOutputNeurons];
		hiddenNeurons_errors = new double[numHiddenNeurons];

		// Get the training cases (if any) from the training file:
		LoadTrainingCases();

	}

	//FUTURE USE [SAVING TO/RESTORING FROM FILE]
	public void LoadTrainingCases() 
	{
		NumTraining = NeuralFile.NumTraining;
		if (NumTraining > 0) 
		{
			inputTraining  = new double[NumTraining * numInputNeurons];
			outputTraining = new double[NumTraining * numOutputNeurons];
		}
		int ic=0, oc=0;

		for (int k=0; k<NumTraining; k++) 
		{
			for (int i=0; i<numInputNeurons; i++)
				inputTraining[ic++] = NeuralFile.GetInput(k, i);
			for (int o=0; o<numOutputNeurons; o++)
				outputTraining[oc++] = NeuralFile.GetOutput(k, o);
		}
	}

 
	//FUTURE USE [SAVING TO/RESTORING FROM FILE]
   void Save(String output_file) {
     if (NeuralFile==null) {
        System.out.println("Error: no NeuralFile object in Neual::Save");
     } else {
        for (int i=0; i<numInputNeurons; i++) {
          for (int h=0; h<numHiddenNeurons; h++) {
            NeuralFile.SetinputToHiddenWeights(i, h, inputToHiddenWeights[i][h]);
          }
        }
        for (int h=0; h<numHiddenNeurons; h++) {
          for (int o=0; o<numOutputNeurons; o++) {
            NeuralFile.SethiddenToOutputWeights(h, o, hiddenToOutputWeights[h][o]);
          }
        }
        NeuralFile.Save(output_file);
     }
   }
*/

}
