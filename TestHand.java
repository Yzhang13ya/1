// TestHand.java
//
// NeuralNet Java classes: tester for handwriting recognition
//

import java.awt.*;

public class TestHand extends GUI implements Runnable 
{

    // define the size of two-dimensional neural
    // input array:
    final static int XSIZE=5;
    final static int YSIZE=6;

    // define the Mode: 0 for training mode,
    // 1 for testing mode:
    public int Mode = 0;

    // Number of characters of each type to use
    // for training data:
    final static int NUM_EX=2;  // number of examples

    // Number of characters to learn:
    final static int NUM = 4;
    String Chars[] = {"a", "b", "c", "d"};;
    // Data for partitioning GUI display for capturing
    // individual characters:
    int X_Pos[] = {20, 60, 100, 140};
    int Y_Pos = 90;
    int Inputs[][][][] = new int[NUM][NUM_EX][XSIZE][YSIZE];

    // Count[NUM] is used for counting drawn chars
    int Count[] = {0, 0, 0, 0};

    // data for determining when a new character
    // is being drawn:
    long TimeLastMouse = -1;  // in milliseconds
    // MouseState: 0=>no capture, 1=>currently capturing
    int MouseState = 0;
    int MousePointIndex = 0; // at the start of capture data


    int num_cap=0;
    int cap_x[] = new int[20000];
    int cap_y[] = new int[20000];

    int active = 0;  // if 1, then train network

    // Neural network:
    Neural network;

    private static final long serialVersionUID=10l;

    public String getAppletInfo() 
    {
        return "Neural Network Simulator for CSCI427";
    }

    private Thread MouseThread = null;

    public void init() 
    {

       Count    = new int[NUM];
       for (int i=0; i<NUM; i++) Count[i] = 0;

       NoInput = true;  // we do not need an input text field
       BigText=1;

       network = new Neural(XSIZE * YSIZE, 10, NUM);
       //network.MyGUI = this;

       TrainLabel = new String("Train");

       super.init();

		if (MouseThread==null) 
		{
        MouseThread = new Thread(this);
        MouseThread.start();
       }
    }

	public void train() 
	{
      P("Starting to train network..wait..\n");
      int sum = 0, ic=0, oc=0;
      for (int i=0; i<NUM; i++)  sum += Count[i];
		  double ins[] = new double[sum*XSIZE*YSIZE];
		  double outs[] = new double[sum*NUM];
      for (int i=0; i<NUM; i++) {
        for (int j=0; j<Count[i]; j++) {
          for (int x=0; x<XSIZE; x++) {
            for (int y=0; y<YSIZE; y++) {
              if (Inputs[i][j][x][y] == 0) {
					ins[ic++] = -0.4;
              }  else  {
					ins[ic++] = +0.4;
            }
          }
        }
        for (int k=0; k<NUM; k++)
			  if (k!=i) outs[oc++] = -0.4;
			  else      outs[oc++] = +0.4;
        }
      }
      for (int i=0; i<3000; i++) {
			double error = network.Train(ins, outs, sum);
        if ((i % 10) == 0) {
           P("Output error for iteration " +
             i + " =" + error + "\n");
        }
			if (error < 0.1)  break;  // done training
      }
    }

    public void run() {
        P("in testHand::run()\n");
        while (true)  {
            try {
               if (MouseState==1) {
                  long mtime =
                    java.lang.System.currentTimeMillis();
                  if (TimeLastMouse < mtime - 800) {
                      MouseState=0;
                      PutChar();
                  }
               }
            }  catch (Exception e) {} ;
            try {Thread.sleep(20);} catch (Exception ex) { };
        }
    }

    public void doTrainButton() 
    {
        train();
        Mode = 1;  // switch to test mode
         P("\nNetwork trained, now in test mode!");
    }

 
    public void PutChar() {
        int x_min=9999, x_max=-9999;
        int y_min=9999, y_max=-9999;
        for (int i=MousePointIndex; i<num_cap; i++) {
            if (cap_x[i] < x_min) x_min = cap_x[i];
            if (cap_x[i] > x_max) x_max = cap_x[i];
            if (cap_y[i] < y_min) y_min = cap_y[i];
            if (cap_y[i] > y_max) y_max = cap_y[i];
        }
        if (x_min+1 > x_max)  { x_min--; x_max++; }
        P("X,Y char bounds: " + x_min + ", " + x_max +
          ", " + y_min + ", " + y_max + "\n");

        // Special case:Mode==1 for testing:
        if (Mode==1) {
            int ic = 0;
            for (int x=0; x<XSIZE; x++) {
                for (int y=0; y<YSIZE; y++) {
                    network.inputNeurons[ic++] = -0.4;
                }
            }
            for (int i=MousePointIndex; i<num_cap; i++) {
                double xx = (double)(cap_x[i] - x_min)
                          / (double)(x_max - x_min);
                xx *= XSIZE;
                double yy = (double)(cap_y[i] - y_min)
                          / (double)(y_max - y_min);
                yy *= YSIZE;
                int ix=(int)xx;
                int iy=(int)yy;
                if (ix<0) ix=0;
                if (ix>=XSIZE) ix=XSIZE-1;
                if (iy<0) iy=0;
                if (iy>=YSIZE) iy=YSIZE-1;
                network.inputNeurons[ix*YSIZE+iy] = +0.4;
            }
            // Propagate input neuron values through
            // to the hidden, then output neuron layer:
            network.ForwardPass();
            // Find the largest output neuron value:
            int index=0;
            double maxVal=-99;
            for (int i=0; i<NUM; i++) {
                if (network.outputNeurons[i]>maxVal) {
                   maxVal = network.outputNeurons[i];
                   index = i;
                }
            }
            P("\nCharacter recognized: " + Chars[index] + "\n");
            repaint();
            return;
        }

        // Find which character is drawn by the x coord:
        int char_type = -1;
        for (int i=0; i<NUM; i++) {
            if (x_min - 10 < X_Pos[i] &&
                x_max + 10 > X_Pos[i]) {
                    char_type = i;
                }
        }
        if (char_type==-1) {
           P("Error: character is not drawn in correct position\n");
           MousePointIndex = num_cap;
           return;
        }
        P("Character " + Chars[char_type] + " drawn. # "
          + Count[char_type] + "\n");
        if (Count[char_type] > (NUM_EX-1)) {
           P("Too many examples for this char type: ignoring!\n");
           MousePointIndex = num_cap;
           return;
        }
        for (int x=0; x<XSIZE; x++) {
            for (int y=0; y<YSIZE; y++) {
                Inputs[char_type][Count[char_type]][x][y] = 0;
            }
        }
        for (int i=MousePointIndex; i<num_cap; i++) {
            double xx = (double)(cap_x[i] - x_min)
                      / (double)(x_max - x_min);
            xx *= XSIZE;
            double yy = (double)(cap_y[i] - y_min)
                      / (double)(y_max - y_min);
            yy *= YSIZE;
            int ix=(int)xx;
            int iy=(int)yy;
            if (ix<0) ix=0;
            if (ix>=XSIZE) ix=XSIZE-1;
            if (iy<0) iy=0;
            if (iy>=YSIZE) iy=YSIZE-1;
            Inputs[char_type][Count[char_type]][ix][iy] = 1;
        }
        MousePointIndex = num_cap;
        Count[char_type] += 1;
    }

    public void doMouseDown(int x, int y) {
        long mtime = java.lang.System.currentTimeMillis();
        if (MouseState==0) { // not yet in capture mode
            P("switch to capture mode\n");
            MouseState=1;
            MousePointIndex = num_cap;
        }

        TimeLastMouse = mtime;

        //System.out.println("Mouse x: " + x + ", y: " + y);
        if (num_cap<19999) {
            cap_x[num_cap] = x;
            cap_y[num_cap] = y;
            num_cap++;
            repaint();
        }
    }
    
    ///////////////////////////////////////////////////////////////////
    // GRAPHICS METHODS
    ///////////////////////////////////////////////////////////////////
	public void paintToDoubleBuffer(Graphics g) {
        g.drawString("Captured handwriting data",
                              X_Pos[0], Y_Pos - 15);

        for (int m=0; m<NUM; m++) 
        {
            g.drawString(Chars[m], X_Pos[m], Y_Pos);
        }
        setForeground(Color.black);
        g.setColor(getForeground());
        for (int i=0; i<num_cap; i++) 
        {
             g.drawLine(cap_x[i],  cap_y[i],  cap_x[i], cap_y[i]+1);
        }
		paintNeuronLayer(g, 170, 100, "Inputs: ", network.inputNeurons, network.numInputNeurons);
		paintNeuronLayer(g, 170, 120, "Hidden: ", network.hiddenNeurons, network.numHiddenNeurons);
		paintNeuronLayer(g, 170, 140, "Outputs: ", network.outputNeurons, network.numOutputNeurons);
		paintWeights(g, 170, 170, "Weights 1: ", network.inputToHiddenWeights, 
												 network.numInputNeurons, network.numHiddenNeurons);
		paintWeights(g, 170, 350, "Weights 2: ", network.hiddenToOutputWeights, 
												 network.numHiddenNeurons, network.numOutputNeurons);

    }

    private void paintNeuronLayer(Graphics g, int x, int y, 
                              String title, 
                              double values[], int num) 
	{
		for (int i=0; i<num; i++) {
			paintGridCell(g, x+60 + i*12, y, 10, 
						  values[i], -0.5, 0.5);
		}
		g.drawString(title, x, y+10);
	}

	private void paintWeights(Graphics g, int x, int y, 
							  String title, double values[][],
							  int num1, int num2) 
	{
		for (int i=0; i<num1; i++) 
		{
			for (int j=0; j<num2; j++) 
			{
				paintGridCell(g, x+80 + i*12, y + j * 12, 10, 
							  values[i][j], -1.5, 1.5);
			}
		}
		g.drawString(title, x, y+10);
	}

}

