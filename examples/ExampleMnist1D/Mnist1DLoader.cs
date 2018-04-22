using System;
using System.Collections.Generic;
using System.IO;
using Zcu.Convsharp.Common;
using Zcu.Convsharp.Loaders;

namespace convsharp.Loaders
{
    /// <summary>
    /// Loader which will load the MNIST dataset from disk as 1D array
    /// http://yann.lecun.com/exdb/mnist/
    /// </summary>
    [Serializable]
    public class Mnist1DLoader : AbstractLoader1D
    {
        /// <summary>
        /// Height of images which will be loaded
        /// </summary>
        private const uint HEIGHT_PX = 28;
        /// <summary>
        /// Width of images which will be loaded
        /// </summary>
        private const uint WIDTH_PX = 28;
        /// <summary>
        /// Max pixel intesity value for normalization
        /// </summary>
        private const uint MAX_PIXEL_VALUE = 255;
        /// <summary>
        /// Maximum number of training data
        /// </summary>
        private const uint TRAIN_DATA_COUNT = 60000;
        /// <summary>
        /// Maximum number of testing data
        /// </summary>
        private const uint TEST_DATA_COUNT = 10000;
        /// <summary>
        /// Number of class in MNIST 0, 1, 2, 3, 4, ... 9
        /// </summary>
        private const uint CLASSES_COUNT = 10;
        /// <summary>
        /// Number of channels in image
        /// </summary>
        private const uint DEPTH_OF_IMAGE = 1;
        /// <summary>
        /// Relative path to training labels
        /// </summary>
        private const string TRAIN_LABELS_RELATIVE_PATH = @"data\mnist\train-labels.idx1-ubyte";
        /// <summary>
        /// Relative path to testing images
        /// </summary>
        private const string TRAIN_IMAGES_RELATIVE_PATH = @"data\mnist\train-images.idx3-ubyte";
        /// <summary>
        /// Relative path to testing labels
        /// </summary>
        private const string TEST_LABELS_RELATIVE_PATH = @"data\mnist\t10k-labels.idx1-ubyte";
        /// <summary>
        /// Relative path to training images
        /// </summary>
        private const string TEST_IMAGES_RELATIVE_PATH = @"data\mnist\t10k-images.idx3-ubyte";
        /// <summary>
        /// Property which if is true data labels will be leaded
        /// as categorical data, otherwise as 2D array (index, label)
        /// </summary>
        private bool categorical = true;
        /// <summary>
        /// Classes of this dataset with ID
        /// </summary>
        private Dictionary<int, uint> classes = null;
        /// <summary>
        /// Absolute path to test images
        /// </summary>
        private string testImagesAbsPath;
        /// <summary>
        /// Absolute path to test labels
        /// </summary>
        private string testLabelsAbsPath;
        /// <summary>
        /// Absolute path to train images
        /// </summary>
        private string trainImagesAbsPath;
        /// <summary>
        /// Absolute path to train labels
        /// </summary>
        private string trainLabelsAbsPath;

        /// <summary>
        /// Constructor for creating new instance of class MLoader
        /// </summary>
        /// <param name="trainItemCount">number of training items
        /// which will be loaded</param>
        /// <param name="testItemCount">number of testing items
        /// which will be loaded</param>
        /// <param name="batchSize">size of batch</param>
        /// <param name="categorical">true if we are expecting categorical data
        /// otherwise false</param>
        public Mnist1DLoader(int trainItemCount, int testItemCount, int batchSize, bool categorical = true)
            : base(trainItemCount, testItemCount, batchSize)
        {
            currentExecuteDirectory = Path.GetDirectoryName(Path.GetDirectoryName(System.IO.Directory.GetCurrentDirectory()));
            testImagesAbsPath = Path.Combine(currentExecuteDirectory, TEST_IMAGES_RELATIVE_PATH);
            testLabelsAbsPath = Path.Combine(currentExecuteDirectory, TEST_LABELS_RELATIVE_PATH);
            trainImagesAbsPath = Path.Combine(currentExecuteDirectory, TRAIN_IMAGES_RELATIVE_PATH);
            trainLabelsAbsPath = Path.Combine(currentExecuteDirectory, TRAIN_LABELS_RELATIVE_PATH);

            this.categorical = categorical;

            if (categorical)
                classes = SetClasses();
        }

        /// <summary>
        /// Set classes according to this will be represented
        /// categorical data e.g. 3 - [0, 0, 1], etc.
        /// </summary>
        /// <returns>dictionary with classes indexes for MNIST</returns>
        private Dictionary<int, uint> SetClasses()
        {
            classes = new Dictionary<int, uint>();
            for (int i = 0; i < CLASSES_COUNT; i++)
                classes.Add(i, (uint)i);
            return classes;
        }

        /// <summary>
        /// Read and saved labels and images from MNIST dataset
        /// </summary>
        /// <param name="imagesAbsPath">Absolute path to images</param>
        /// <param name="labelsAbsPath">Absolute path to labels</param>
        /// <param name="train">Expected training data if is true, otherwise false</param>
        /// <param name="samples">Number of pictures which will be returned</param>
        /// <returns>Array with images and labels saved in Tuple</returns>
        private Tuple<double[], double[]> ReadLabelAndImage(string imagesAbsPath, string labelsAbsPath, int index, bool train)
        {
            // maximum number of samples
            uint dataCount = train ? TRAIN_DATA_COUNT : TEST_DATA_COUNT;
            // array for image
            double[] image = null;
            double[] labels = null;

            try
            {
                // images and labesls stream
                FileStream streamImages = new FileStream(imagesAbsPath, FileMode.Open);
                FileStream streamLabels = new FileStream(labelsAbsPath, FileMode.Open);
                // readers for idx files (we will go byte by byte)
                BinaryReader readerImages = new BinaryReader(streamImages);
                BinaryReader readerLabels = new BinaryReader(streamLabels);
                // discard header of images file
                readerImages.ReadInt32();
                readerImages.ReadInt32();
                readerImages.ReadInt32();
                readerImages.ReadInt32();
                // discard header of labels file
                readerLabels.ReadInt32();
                readerLabels.ReadInt32();
                // init image array
                int[] labelsInt = new int[1];
                byte value;
                double normValue;

                readerImages.BaseStream.Seek(WIDTH_PX * HEIGHT_PX * index, SeekOrigin.Current);
                readerLabels.BaseStream.Seek(index, SeekOrigin.Current);

                // create pixel arrays according to size of pictures
                image = new double[HEIGHT_PX * WIDTH_PX * DEPTH_OF_IMAGE];
                
                for (int i = 0; i < HEIGHT_PX; i++)
                {
                    for (int j = 0; j < WIDTH_PX; j++)
                    {
                        // read value
                        value = readerImages.ReadByte();
                        // normalize value to 0 - 1
                        normValue = ((double)value) / MAX_PIXEL_VALUE;
                        // save into picture matrix
                        image[(i * HEIGHT_PX) + j] = normValue;
                    }
                }

                // read label
                byte label = readerLabels.ReadByte();
                labelsInt[0] = label;

                // convert labels into categorical matrix
                if (categorical)
                {
                    Tuple<double[][], Dictionary<int, uint>> res = Utils.ToCategorical(labelsInt, CLASSES_COUNT, classes);
                    labels = res.Item1[0];
                    classes = res.Item2;
                }
                else
                {
                    labels = new double[1];
                    labels[0] = labelsInt[0];
                }

                streamImages.Dispose();
                streamLabels.Dispose();

            }
            catch (Exception ex)
            {
                string msg = "Loading MNIST dataset failed with error message " + ex.Message;
                Utils.ThrowException(msg);
            }

            return new Tuple<double[], double[]>(image, labels);
        }

        public override Tuple<double[], double[]> Load(int itemIndex, bool train)
        {
            string imagePath = train ? trainImagesAbsPath : testImagesAbsPath;
            string labelsPath = train ? trainLabelsAbsPath : testLabelsAbsPath;
            return ReadLabelAndImage(imagePath, labelsPath, itemIndex, train);
        }
    }
}