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
    /// Simple multilayer neural network for Iris dataset
    /// recognizing https://archive.ics.uci.edu/ml/datasets/iris
    /// Problem is quite simple so we will use MLP with hidden layer
    /// with 16 neurons stochastic gradient descent and categorical
    /// cross-entropy as loss function.
    /// </summary>
    public class ExampleIris
    {
        public static void Main(string[] args)
        {
            int batchSize = 16;

            // as input dimension we will use only four features
            Dimension inputDim = new Dimension(batchSize, 1, 1, 4);

            // architecture of network
            FlattenLayer flatten = new FlattenLayer(inputDim);
            LinearLayer linear0 = new LinearLayer(numNeurons: 32);
            ActivationLayer activation0 = new ActivationLayer(new Relu());
            LinearLayer linear1 = new LinearLayer(numNeurons: 3);
            ActivationLayer activation1 = new ActivationLayer(new Softmax());

            SequentialModel model = new SequentialModel();
            model.Add(flatten);
            model.Add(linear0);
            model.Add(activation0);
            model.Add(linear1);
            model.Add(activation1);

            // compile modela
            model.Compile(new CategoricalCrossEntropy(), new Adam(0.001d));

            // train the model
            List<EpochHistory> history = model.Fit(new IrisLoader(120, 30, batchSize: batchSize), epochCount: 12, useValidationSet: true);

            // show architecture of MLP with params
            model.Summary();
        }
    }
}