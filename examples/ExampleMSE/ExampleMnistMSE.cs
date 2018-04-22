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
    /// Example of non categorical data used as input.
    /// Architecture is same only instead of categorical
    /// cross-entropy we will use mean squared error.
    /// Compressed dataset: http://yann.lecun.com/exdb/mnist/
    /// </summary>
    public class ExampleMnistMSE
    {
        public static void Main(string[] args)
        {
            int batchSize = 32;

            // Size of input dimension
            Dimension inputDim = new Dimension(batchSize, 1, 28, 28);
            
            // Architecture of CNN
            Convolution2DLayer conv = new Convolution2DLayer(inputDim, filterSize: 3, filterCount: 20, zeroPadding: false);
            MaxPooling2DLayer pool = new MaxPooling2DLayer();
            FlattenLayer flatten = new FlattenLayer();
            LinearLayer linear0 = new LinearLayer(numNeurons: 64);
            ActivationLayer activation = new ActivationLayer(new Relu());
            LinearLayer linear1 = new LinearLayer(numNeurons: 1);

            // Create new sequential model and adding layers
            SequentialModel model = new SequentialModel();
            model.Add(conv);
            model.Add(pool);
            model.Add(flatten);
            model.Add(linear0);
            model.Add(activation);
            model.Add(linear1);

            // Compile model
            model.Compile(new MeanSquaredError(), new Adam(0.001d));

            // Train model and use validation set for testing
            // we will use 1000 training and 100 testing images
            List<EpochHistory> history = model.Fit(new MnistLoader(1000, 100, batchSize: batchSize, categorical: false), 
                epochCount: 10, useValidationSet: false);

            // Show architecture of CNN with params
            model.Summary();
        }
    }
}