package org.deeplearning4j.examples.experiments;

import org.deeplearning4j.eval.Evaluation;
import org.deeplearning4j.nn.api.Layer;
import org.deeplearning4j.nn.api.OptimizationAlgorithm;
import org.deeplearning4j.nn.conf.MultiLayerConfiguration;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration.ListBuilder;
import org.deeplearning4j.nn.conf.distribution.UniformDistribution;
import org.deeplearning4j.nn.conf.layers.DenseLayer;
import org.deeplearning4j.nn.conf.layers.OutputLayer.Builder;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.nn.weights.WeightInit;
import org.deeplearning4j.optimize.listeners.ScoreIterationListener;
import org.deeplearning4j.ui.weights.HistogramIterationListener;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.lossfunctions.LossFunctions;

/**
 * This basic example shows how to manually create a DataSet and train it to an
 * basic Network.
 * <p/>
 * The network consists in 2 input-neurons, 1 hidden-layer with 4
 * hidden-neurons, and 2 output-neurons.
 * <p/>
 * I choose 2 output neurons, (the first fires for false, the second fires for
 * true) because the Evaluation class needs one neuron per classification.
 *
 * @author Peter Großmann
 */
public class FizzBuzz {

    public static int INPUT_LAYER_NEURONS = 4;
    public static final int NR_HIDDEN_LAYERS = 2;
    public static final int HIDDEN_NEURONS = 2;
    public static int OUTPUT_LAYER_NEURONS = 2;

    public static final int OUTPUT_CLASSES = 4;
    public static final int NUM_ITERATIONS = 1000;


    public static final int MAX_NUMBER = 10000;

    public static void main(String[] args) {

        INPUT_LAYER_NEURONS = Integer.toBinaryString(MAX_NUMBER).length();

        int examplePerc = MAX_NUMBER;

        INDArray input = Nd4j.zeros(examplePerc, INPUT_LAYER_NEURONS);
        INDArray labels = Nd4j.zeros(examplePerc, OUTPUT_LAYER_NEURONS);

        for (int i = 0; i < examplePerc; i++) {
            String bits = Integer.toBinaryString(i);
            int bitDifference = INPUT_LAYER_NEURONS - bits.length();

            for (int j = bitDifference; j < INPUT_LAYER_NEURONS; j++) {
                input.putScalar(new int[]{i, j}, Character.getNumericValue(bits.charAt(j - bitDifference))); // fill with the rest
            }

            if (i % 15 == 0) {
                labels.putScalar(new int[]{i, 0}, 1);
                labels.putScalar(new int[]{i, 1}, 1);
            } else if (i % 5 == 0) {
                labels.putScalar(new int[]{i, 0}, 1);
                labels.putScalar(new int[]{i, 1}, 0);
            } else if (i % 3 == 0) {
                labels.putScalar(new int[]{i, 0}, 0);
                labels.putScalar(new int[]{i, 1}, 1);
            } else {
                labels.putScalar(new int[]{i, 0}, 0);
                labels.putScalar(new int[]{i, 1}, 0);
            }

        }

        // create dataset object
        DataSet ds = new DataSet(input, labels);

        // Set up network configuration
        NeuralNetConfiguration.Builder builder = new NeuralNetConfiguration.Builder();
        // how often should the training set be run, we need something above
        // 1000, or a higher learning-rate - found this values just by trial and
        // error
        builder.iterations(NUM_ITERATIONS);
        // learning rate
        builder.learningRate(0.1);
        // fixed seed for the random generator, so any run of this program
        // brings the same results - may not work if you do something like
        // ds.shuffle()
        builder.seed(123);
        // not applicable, this network is to small - but for bigger networks it
        // can help that the network will not only recite the training data
        builder.useDropConnect(false);
        // a standard algorithm for moving on the error-plane, this one works
        // best for me, LINE_GRADIENT_DESCENT or CONJUGATE_GRADIENT can do the
        // job, too - it's an empirical value which one matches best to
        // your problem
        builder.optimizationAlgo(OptimizationAlgorithm.LINE_GRADIENT_DESCENT);
        // init the bias with 0 - empirical value, too
        builder.biasInit(0);
        // from "http://deeplearning4j.org/architecture": The networks can
        // process the input more quickly and more accurately by ingesting
        // minibatches 5-10 elements at a time in parallel.
        // this example runs better without, because the dataset is smaller than
        // the mini batch size
        builder.miniBatch(true);

        // create a multilayer network with 2 layers (including the output
        // layer, excluding the input payer)

        ListBuilder listBuilder = builder.list(NR_HIDDEN_LAYERS + 1);

        buildFirstHiddenLayer(listBuilder);
        buildHiddenLayers(listBuilder);
        buildOuputLayer(listBuilder);


        // no pretrain phase for this network
        listBuilder.pretrain(false);

        // seems to be mandatory
        // according to agibsonccc: You typically only use that with
        // pretrain(true) when you want to do pretrain/finetune without changing
        // the previous layers finetuned weights that's for autoencoders and
        // rbms
        listBuilder.backprop(true);

        // build and init the network, will check if everything is configured
        // correct
        MultiLayerConfiguration conf = listBuilder.build();
        MultiLayerNetwork net = new MultiLayerNetwork(conf);
        net.init();
        net.setListeners(new HistogramIterationListener(1));

        // add an litener which outputs the error every 10 samples
        net.setListeners(new ScoreIterationListener(10));

        // C&P from GravesLSTMCharModellingExample
        // Print the number of parameters in the network (and for each layer)
        Layer[] layers = net.getLayers();
        int totalNumParams = 0;
        for (int i = 0; i < layers.length; i++) {
            int nParams = layers[i].numParams();
            System.out.println("Number of parameters in layer " + i + ": " + nParams);
            totalNumParams += nParams;
        }
        System.out.println("Total number of network parameters: " + totalNumParams);

        // here the actual learning takes place
        net.fit(ds);

        // create output for every training sample
        INDArray output = net.output(ds.getFeatureMatrix());
        System.out.println(output);

        // let Evaluation prints stats how often the right output had the
        // highest value
        Evaluation eval = new Evaluation(OUTPUT_CLASSES);
        eval.eval(ds.getLabels(), output);
        System.out.println(eval.stats());

//        int howManyToTestOn = 100;
//        int start = 30;
//        INDArray whatToP = Nd4j.zeros(howManyToTestOn, INPUT_LAYER_NEURONS);
//        for (int i = start; i < howManyToTestOn; i++) {
//            String bits = Integer.toBinaryString(i);
//            int bitDifference = INPUT_LAYER_NEURONS - bits.length();
//
//            for (int j = bitDifference; j < INPUT_LAYER_NEURONS; j++) {
//                whatToP.putScalar(new int[]{i - start, j}, Character.getNumericValue(bits.charAt(j - bitDifference))); // fill with the rest
//            }
//        }
//
//        int[] predict = net.predict(whatToP);
//        for (int i = 1; i < howManyToTestOn; i++) {
//            switch (predict[i]) {
//                case FIZZ:
//                    out.println(whatToP.getScalar(i) + " - FIZZ");
//                    break;
//                case BUZZ:
//                    out.println(whatToP.getScalar(i) + " - BUZZ");
//                    break;
//                case FIZZBUZZ:
//                    out.println(whatToP.getScalar(i) + " - FIZZBUZZ");
//                    break;
//                case NONE:
//                    out.println(whatToP.getScalar(i) + " - NONE");
//                    break;
//            }
//        }

    }

    private static void buildOuputLayer(ListBuilder listBuilder) {
        // MCXENT or NEGATIVELOGLIKELIHOOD work ok for this example - this
        // function calculates the error-value
        // From homepage: Your net's purpose will determine the loss funtion you
        // use. For pretraining, choose reconstruction entropy. For
        // classification, use multiclass cross entropy.
        Builder outputLayerBuilder = new Builder(LossFunctions.LossFunction.NEGATIVELOGLIKELIHOOD);
        // must be the same amout as neurons in the layer before
        outputLayerBuilder.nIn(HIDDEN_NEURONS);
        // two neurons in this layer
        outputLayerBuilder.nOut(2);
        outputLayerBuilder.activation("sigmoid");
        outputLayerBuilder.weightInit(WeightInit.DISTRIBUTION);
        outputLayerBuilder.dist(new UniformDistribution(0, 1));
        listBuilder.layer(NR_HIDDEN_LAYERS, outputLayerBuilder.build());
    }

    private static void buildHiddenLayers(ListBuilder listBuilder) {
        for (int i = 1; i < NR_HIDDEN_LAYERS; i++) {
            DenseLayer.Builder hiddenLayerBuilder = new DenseLayer.Builder();
            // two input connections - simultaneously defines the number of input
            // neurons, because it's the first non-input-layer
            hiddenLayerBuilder.nIn(HIDDEN_NEURONS);
            // number of outgooing connections, nOut simultaneously defines the
            // number of neurons in this layer
            hiddenLayerBuilder.nOut(HIDDEN_NEURONS);
            // put the output through the sigmoid function, to cap the output
            // valuebetween 0 and 1
            hiddenLayerBuilder.activation("sigmoid");
            // random initialize weights with values between 0 and 1
            hiddenLayerBuilder.weightInit(WeightInit.DISTRIBUTION);
            hiddenLayerBuilder.dist(new UniformDistribution(0, 1));
            listBuilder.layer(i, hiddenLayerBuilder.build());
        }
    }

    private static void buildFirstHiddenLayer(ListBuilder listBuilder) {
        DenseLayer.Builder hiddenLayerBuilder = new DenseLayer.Builder();
        // two input connections - simultaneously defines the number of input
        // neurons, because it's the first non-input-layer
        hiddenLayerBuilder.nIn(INPUT_LAYER_NEURONS);
        // number of outgooing connections, nOut simultaneously defines the
        // number of neurons in this layer
        hiddenLayerBuilder.nOut(HIDDEN_NEURONS);
        // put the output through the sigmoid function, to cap the output
        // valuebetween 0 and 1
        hiddenLayerBuilder.activation("sigmoid");
        // random initialize weights with values between 0 and 1
        hiddenLayerBuilder.weightInit(WeightInit.DISTRIBUTION);
        hiddenLayerBuilder.dist(new UniformDistribution(0, 1));
        // build and set as layer 0
        listBuilder.layer(0, hiddenLayerBuilder.build());
    }
}
