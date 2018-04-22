using convsharp.Loaders;
using System.Collections.Generic;
using Zcu.Convsharp.Common;
using Zcu.Convsharp.CostFunctions;
using Zcu.Convsharp.Layer;
using Zcu.Convsharp.Layers.ActivationFunctions;
using Zcu.Convsharp.Model;
using Zcu.Convsharp.Optimizers;

namespace Zcu.Convsharp
{
    /// <summary>
    /// As example for usage 1D convolution and pooling 
    /// we will use MNIST dataset where are all pixels
    /// in one dimension. We will use same architecture
    /// of model as in ExampleMnist.cs file but instead
    /// of 2D convolution and pooling we will use 1D.
    /// Compressed dataset: http://yann.lecun.com/exdb/mnist/
    /// </summary>
    public class ExampleMnist1D
    {
        public static void Main(string[] args)
        {
            int batchSize = 32;

            Dimension inputDim = new Dimension(batchSize, 1, 1, 28 * 28);

            // Architecture of CNN
            Convolution1DLayer conv = new Convolution1DLayer(inputDim, filterSize: 3, filterCount: 20, zeroPadding: false);
            MaxPooling1DLayer pool = new MaxPooling1DLayer();
            FlattenLayer flatten = new FlattenLayer();
            LinearLayer linear0 = new LinearLayer(numNeurons: 64);
            ActivationLayer activation0 = new ActivationLayer(new Relu());
            LinearLayer linear1 = new LinearLayer(numNeurons: 10);
            ActivationLayer activation1 = new ActivationLayer(new Softmax());

            // Create new sequential model and adding layers
            SequentialModel model = new SequentialModel();
            model.Add(conv);
            model.Add(pool);
            model.Add(flatten);
            model.Add(linear0);
            model.Add(activation0);
            model.Add(linear1);
            model.Add(activation1);

            // Compile model
            model.Compile(new CategoricalCrossEntropy(), new Adam(0.001d));

            // Train model and use validation set for testing
            // we will use 1000 training and 100 testing images
            List<EpochHistory> history = model.Fit(new Mnist1DLoader(1000, 100, batchSize: batchSize), epochCount: 12, useValidationSet: true);

            // Show architecture of CNN with params
            model.Summary();           
        }
    }
}