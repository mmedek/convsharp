using System.Collections.Generic;
using System.IO;
using Zcu.Convsharp.Common;
using Zcu.Convsharp.CostFunctions;
using Zcu.Convsharp.Layer;
using Zcu.Convsharp.Layers.ActivationFunctions;
using Zcu.Convsharp.Model;
using Zcu.Convsharp.Optimizers;

namespace Zcu.Convsharp
{
    /// <summary>
    /// Example of binary problem solved by CNN
    /// We will use CNN with two convolution layers
    /// followed by activation and max-pooling layer.
    /// We will use ReLU as activation function in
    /// hidden layers and sigmoid with binary cross
    /// entropy in output layer. As optimalization
    /// algorithm we will use ADAM.
    /// </summary>
    public class ExampleDogsCats
    {

        public static void Main(string[] args)
        {
            // Images are quite big 128x128 pixels so we will
            // use small batches
            int batchSize = 32;

            // Input dimension is 128 x 128 x 1 (width x height x depth)
            // we will use grayscaled images
            Dimension inputDim = new Dimension(batchSize, 3, 128, 128);

            // architecture of CNN
            Convolution2DLayer conv0 = new Convolution2DLayer(inputDim, filterSize: 3, filterCount: 32, zeroPadding: true);
            ActivationLayer activation0 = new ActivationLayer(new Relu());
            MaxPooling2DLayer pool0 = new MaxPooling2DLayer();
            Convolution2DLayer conv1 = new Convolution2DLayer(inputDim, filterSize: 3, filterCount: 64, zeroPadding: true);
            ActivationLayer activation1 = new ActivationLayer(new Relu());
            MaxPooling2DLayer pool1 = new MaxPooling2DLayer();
            FlattenLayer flatten = new FlattenLayer();
            LinearLayer linear0 = new LinearLayer(numNeurons: 128);
            ActivationLayer activation2 = new ActivationLayer(new Relu());
            LinearLayer linear1 = new LinearLayer(numNeurons: 1);
            ActivationLayer activation3 = new ActivationLayer(new Sigmoid());

            // create new sequential model and adding layers
            SequentialModel model = new SequentialModel();
            model.Add(conv0);
            model.Add(activation0);
            model.Add(pool0);
            model.Add(conv1);
            model.Add(activation1);
            model.Add(pool1);
            model.Add(flatten);
            model.Add(linear0);
            model.Add(activation2);
            model.Add(linear1);
            model.Add(activation3);

            // compile model
            model.Compile(new BinaryCrossEntropy(), new Adam(0.001d));

            // set up custom loader
            // folders with images are placed in data folder which
            // is placed like exe file
            // in this case you have to download data from the kaggle
            // if you want to run this example
            string currentExecuteDirectory = Path.GetDirectoryName(System.Reflection.Assembly.GetEntryAssembly().Location);
            string trainImagesAbsPath = Path.Combine(currentExecuteDirectory, @"data\dogs_vs_cats_train\");
            string testImagesAbsPath = Path.Combine(currentExecuteDirectory, @"data\dogs_vs_cats_test\");
            // we will use 1000 training and 100 testing images
            DogsCatsLoader loader = new DogsCatsLoader(1000, 100, batchSize: batchSize,
                trainPath: trainImagesAbsPath, testPath: testImagesAbsPath);
            
            // train model and use validation set for testing
            List<EpochHistory> history = model.Fit(loader, epochCount: 100, useValidationSet: true);

            // show architecture of CNN with params
            model.Summary();
        }
    }
}