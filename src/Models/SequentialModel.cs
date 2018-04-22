using System;
using System.Collections.Generic;
using System.IO;
using Zcu.Convsharp.Common;
using Zcu.Convsharp.CostFunctions;
using Zcu.Convsharp.Layer;
using Zcu.Convsharp.Loaders;
using Zcu.Convsharp.Logger;
using Zcu.Convsharp.Optimizers;

namespace Zcu.Convsharp.Model
{
    /// <summary>
    /// Implementation of sequential model for neural network
    /// </summary>
    [Serializable]
    public class SequentialModel : AbstractModel
    {
        /// <summary>
        /// List of layers
        /// </summary>
        private List<AbstractLayer> layers;
        /// <summary>
        /// Flag which is true if model was already
        /// trained
        /// </summary>
        private bool alreadyTrained = false;
        /// <summary>
        /// Loss function of this model
        /// </summary>
        private AbstractCostFunction lossFunction;
        /// <summary>
        /// Optimizer of this model
        /// </summary>
        private AbstractOptimizer optimizer;

        /// <summary>
        /// Constructor for creating new instance of 
        /// SequentialModel
        /// </summary>
        public SequentialModel()
        {
            layers = new List<AbstractLayer>();
        }

        public override void Add(AbstractLayer layer)
        {
            layers.Add(layer);
        }

        public override void Compile(AbstractCostFunction lossFunction, AbstractOptimizer optimizer)
        {
            AbstractLayer previousLayer = null;

            // compile each layer and check dimension
            // of input and output
            foreach (AbstractLayer layer in layers)
            {
                previousLayer = layer.Compile(previousLayer);
            }

            this.lossFunction = lossFunction;
            this.optimizer = optimizer;
        }

        public override int Predict(double[][][] item)
        {
            double[][][][] temp = new double[1][][][];
            temp[0] = item;
            double[][][][] result = ComputeOutput(temp);
            return lossFunction.GetResult(result[0]);
        }

        public override void Summary()
        {
            if (!alreadyTrained)
            {
                Logger.Log.Warning("It is not possible to display" +
                    " summary of the model. Please, train the" +
                    " model first.");
                return;
            }
            string[] layerSummary = new string[3];
            long paramsSum = 0;
            string output = String.Format("{0,-15} {1, -30} {2, -45}\n", "Layer Type", "Output Shape", "Params #");
            Log.Info(output);
            for (int i = 0; i < layers.Count; i++)
            {
                paramsSum += layers[i].Summary();
            }
            output = String.Format("\nTotal parameters: " + paramsSum);
            Log.Info(output);
        }

        public override Tuple<double, double> Evaluate(double[][][][] testImages, double[][] testLabels, bool print = true)
        {
            double[][][][] output = ComputeOutput(testImages);
            double loss = lossFunction.Compute(output, testLabels);
            for (int i = 0; i < layers.Count; i++)
            {
                if (layers[i] is ILearnable && ((ILearnable)layers[i]).Regularizer != null)
                {
                    ILearnable learnableLayer = (ILearnable)layers[i];
                    int numSamples = learnableLayer.Weights.Length;
                    loss += learnableLayer.Regularizer.ComputeAdditionToCostFunc(learnableLayer.Weights, numSamples);
                }
            }

            double acc = lossFunction.ComputeAccuracy(output, testLabels);
            if (print)
                Logger.Log.Info("Test accuracy is " + acc.ToString("0.####"));
            return Tuple.Create<double, double>(acc, loss);
        }

        public override void Save(string pathToModel)
        {
            try
            {
                FileStream stream = File.Create(pathToModel);
                var formatter = new System.Runtime.Serialization.Formatters.Binary.BinaryFormatter();
                formatter.Serialize(stream, this);
                stream.Close();
            }
            catch(Exception ex)
            {
                Utils.ThrowException("Saving model failed with following error message " + ex.Message);
            }
        }

        public override List<EpochHistory> Fit(AbstractLoader loader, int epochCount, bool useValidationSet = false)
        {
            if (!IsAlreadyCompiled())
            {
                Logger.Log.Warning("It is not possible to train" +
                    " the model before compiling model.");
                return new List<EpochHistory>();
            }

            alreadyTrained = true;

            double[][][][] currentInput;
            double[][][][] currentGradient;
            double[][] trainLabels;
            string outputstring = "";

            // proceed number of epochs
            List<EpochHistory> history = new List<EpochHistory>();
            for (int epochCounter = 0; epochCounter < epochCount; epochCounter++)
            {
                // proceed number of iterations according to batches
                List<Tuple<double, double>> trainResults = new List<Tuple<double, double>>();
                for (int batchIndex = 0; batchIndex < loader.TrainBatchCount; batchIndex++)
                {
                    var batch = loader.LoadBatch(batchIndex, train: true);
                    currentInput = batch.Item1;
                    trainLabels = batch.Item2;

                    // proceed feed forward step
                    for (int i = 0; i < layers.Count; i++)
                    {
                        currentInput = layers[i].ForwardPropagation(currentInput);
                    }
                  
                    // output layer gradient
                    currentGradient = lossFunction.Derivate(currentInput, trainLabels);

                    // backward procedure
                    for (int i = layers.Count - 1; i >= 0; i--)
                    {
                        currentGradient = layers[i].BackwardPropagation(currentGradient);
                    }

                    // update weights
                    for (int i = 0; i < layers.Count; i++)
                    {
                        if (layers[i] is ILearnable)
                            optimizer.UpdateWeights((ILearnable)layers[i], (epochCounter + 1));
                    }

                    trainResults.Add(Evaluate(batch.Item1, trainLabels, print: false));
                }

                // compute average accuracy and loss for each
                // iteration
                Tuple<double, double> trainResult = ComputeFinalTrainAcc(trainResults);
                outputstring = "Epoch " + (epochCounter + 1).ToString() + "/" + epochCount
                    + " " + loader.TrainItemCount + " samples " 
                    + "- train_loss: " + trainResult.Item2.ToString("0.####")
                    + " - train_acc: " + trainResult.Item1.ToString("0.####");
                
                // if we are using validation set
                // compute accuracy and loss for
                // this set too
                if (useValidationSet)
                {
                    List<Tuple<double, double>> testResults = new List<Tuple<double, double>>();
                    for (int batchIndex = 0; batchIndex < loader.TestBatchCount; batchIndex++)
                    {
                        var batch = loader.LoadBatch(batchIndex, train: false);
                        testResults.Add(Evaluate(batch.Item1, batch.Item2, print: false));
                    }

                    Tuple<double, double> testResult = ComputeFinalTrainAcc(testResults);
                    
                    history.Add(new EpochHistory(testResult.Item2, trainResult.Item2, testResult.Item1, trainResult.Item1));

                    outputstring += " - test_loss: " + testResult.Item2.ToString("0.####")
                        + " - test_acc: " + testResult.Item1.ToString("0.####");
                }
                else
                {
                    history.Add(new EpochHistory(0d, trainResult.Item2, 0d, trainResult.Item1));
                }

                Logger.Log.Info(outputstring);
            }

            return history;
        }

        /// <summary>
        /// Proceed forward propagation through all
        /// layers according to input
        /// </summary>
        /// <param name="originalInput">Input into network</param>
        /// <returns>Result from output layer</returns>
        private double[][][][] ComputeOutput(double[][][][] originalInput)
        {
            double[][][][] currentOutput = originalInput;
            for (int i = 0; i < layers.Count; i++)
            {
                currentOutput = layers[i].ForwardPropagation(currentOutput, predict: true);
            }
            return currentOutput;
        }

        /// <summary>
        /// Method sum up loss and accuracy computed during
        /// each iteration and divide it with the iteration count
        /// </summary>
        /// <param name="trainResults">Accuracy and loss
        /// computed during each iteration</param>
        /// <returns>tuple (average accuracy, average loss)</returns>
        private Tuple<double, double> ComputeFinalTrainAcc(List<Tuple<double, double>> trainResults)
        {
            double length = trainResults.Count;
            double mult = 1 / length;

            double sumAcc = 0;
            double sumLoss = 0;

            // sum loss and accuracy
            // and normalize with
            // iteration count
            for (int i = 0; i < length; i++)
            {
                sumAcc += trainResults[i].Item1;
                sumLoss += trainResults[i].Item2;
            }

            return Tuple.Create(sumAcc * mult, sumLoss * mult);
        }

        /// <summary>
        /// Method which check if model
        /// was already compiled.
        /// </summary>
        /// <returns>true if it was compiled, otherwise false</returns>
        private bool IsAlreadyCompiled()
        {
            for (int i = 0; i < layers.Count; i++)
            {
                if (!layers[0].Compiled)
                    return false;
            }
            return true;
        }
    }
}