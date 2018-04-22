using System;
using System.Collections.Generic;
using Zcu.Convsharp.Common;
using Zcu.Convsharp.CostFunctions;
using Zcu.Convsharp.Layer;
using Zcu.Convsharp.Layers.ActivationFunctions;
using Zcu.Convsharp.Loaders;
using Zcu.Convsharp.Model;
using Zcu.Convsharp.Optimizers;

namespace Zcu.Convsharp
{
    /// <summary>
    /// Example of simple CNN which will classify MNIST dataset
    /// We will use convolution layer followed by ReLU activation
    /// and maxpooling layer. After that we will use flatten layer
    /// and linear layer with 64 neurons and ReLU activation.
    /// In the output layer is placed 10 neurons with softmax
    /// function.
    /// Compressed dataset: http://yann.lecun.com/exdb/mnist/
    /// </summary>
    public class ExampleMnist
    {
        public static void Main(string[] args)
        {
            int batchSize = 32;

            // input dimension 1 = depth, 28 = image height, 28 = image width
            Dimension inputDim = new Dimension(batchSize, 1, 28, 28);

            // Architecture of CNN
            Convolution2DLayer conv = new Convolution2DLayer(inputDim, filterSize: 3, filterCount: 20, zeroPadding: false);
            ActivationLayer activation0 = new ActivationLayer(new Relu());
            MaxPooling2DLayer pool = new MaxPooling2DLayer();
            FlattenLayer flatten = new FlattenLayer();
            LinearLayer linear = new LinearLayer(numNeurons: 64);
            ActivationLayer activation = new ActivationLayer(new Relu());
            LinearLayer linear2 = new LinearLayer(numNeurons: 10);
            ActivationLayer activation2 = new ActivationLayer(new Softmax());

            // Create new sequential model and adding layers
            SequentialModel model = new SequentialModel();
            model.Add(conv);
            model.Add(activation0);
            model.Add(pool);
            model.Add(flatten);
            model.Add(linear);
            model.Add(activation);
            model.Add(linear2);
            model.Add(activation2);

            // Compile model
            model.Compile(new CategoricalCrossEntropy(), new Adam(0.001d));

            // Train model and use validation set for testing
            // we will use 1000 training and 100 testing images
            MnistLoader loader = new MnistLoader(1000, 100, batchSize: batchSize);

            List<EpochHistory> history = model.Fit(loader, epochCount: 4, useValidationSet: true);

            // Example of simple prediction for one item
            var batch = loader.LoadBatch(0, false);
            double[][][] firstItem = batch.Item1[0];
            // Basic visualization of image
            Visualize(firstItem);
            int result = model.Predict(firstItem);
            Console.WriteLine("Model classify image as class '" + result + "'");

            // Show architecture of CNN with params
            model.Summary();

            string pathToModel = @"savedModel.dat";
            // Save model
            model.Save(pathToModel);

            // Load model again
            SequentialModel loadedModel = Utils.LoadModel(pathToModel);
        }

        private static void Visualize(double[][][] v1)
        {
            for (int x = 0; x < v1[0].Length; x++)
            {
                for (int y = 0; y < v1[0][0].Length; y++)
                {
                    Console.Write((v1[0][x][y] > 0.35d) ? "o" : "X");
                }
                Console.WriteLine();
            }
        }
    }
}