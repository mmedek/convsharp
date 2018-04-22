using System;
using System.Collections.Generic;
using System.IO;
using Zcu.Convsharp.Common;
using Zcu.Convsharp.Loaders;

namespace convsharp.Loaders
{
    /// <summary>
    /// Loader which will load the IRIS dataset from csv files on disk
    /// https://archive.ics.uci.edu/ml/datasets/iris
    /// </summary>
    [Serializable]
    public class IrisLoader : AbstractLoader1D
    {
        /// <summary>
        /// Maximum number of training data
        /// </summary>
        private const uint TRAIN_DATA_COUNT = 120;
        /// <summary>
        /// Maximum number of testing data
        /// </summary>
        private const uint TEST_DATA_COUNT = 30;
        /// <summary>
        /// Number of class in IRIS 0, 1, 2
        /// </summary>
        private const uint CLASSES_COUNT = 3;
        /// <summary>
        /// Relative path to training images
        /// </summary>
        private const string TRAIN_IMAGES_RELATIVE_PATH = @"data\iris_train.csv";
        /// <summary>
        /// Relative path to testing images
        /// </summary>
        private const string TEST_IMAGES_RELATIVE_PATH = @"data\iris_test.csv";
        /// <summary>
        /// Classes of this dataset with ID
        /// </summary>
        private Dictionary<int, uint> classes = null;
        /// <summary>
        /// Variable which is is true we are return categorical data
        /// otherwise non categorical data
        /// </summary>
        private bool categorical;
        /// <summary>
        /// List which holds each row of loaded csv file
        /// for training data
        /// </summary>
        private string[,] trainValues;
        /// <summary>
        /// List which holds each row of loaded csv file
        /// for testing data
        /// </summary>
        private string[,] testValues;
        /// <summary>
        /// Number of columns in csv from which are
        /// data loaded
        /// </summary>
        private const int COLS = 5;

        /// <summary>
        /// Constructor for creating new instance of class IrisLoader
        /// </summary>
        /// <param name="trainItemCount">number of training items
        /// which will be loaded</param>
        /// <param name="testItemCount">number of testing items
        /// which will be loaded</param>
        /// <param name="batchSize">size of batch</param>
        /// <param name="categorical">true if we are expecting categorical data
        /// otherwise false</param>
        public IrisLoader(int trainItemCount, int testItemCount, int batchSize, bool categorical = true)
            : base(trainItemCount, testItemCount, batchSize)
        {

            if (trainItemCount > TRAIN_DATA_COUNT || testItemCount > TEST_DATA_COUNT || trainItemCount < 1)
                Utils.ThrowException("Invalid number of items in IrisLoader");

            currentExecuteDirectory = Path.GetDirectoryName(System.Reflection.Assembly.GetEntryAssembly().Location);

            // Allocation space for trianing data
            trainValues = new string[trainItemCount, COLS];

            // Allocation space for testing data
            testValues = new string[testItemCount, COLS];

            // Load training and testing data from csv
            PrepareData(trainItemCount, TRAIN_IMAGES_RELATIVE_PATH, true);
            PrepareData(testItemCount, TEST_IMAGES_RELATIVE_PATH, false);

            this.categorical = categorical;

            if (categorical)
                classes = SetClasses();
        }

        private void PrepareData(int numberOfSamples, string path, bool train)
        {
            // Build path to csv files
            string testLabelsRelativePath = path;
            string currentExecuteDirectory = Path.GetDirectoryName(Path.GetDirectoryName(System.IO.Directory.GetCurrentDirectory()));
            string trainImagesAbsPath = Path.Combine(currentExecuteDirectory, testLabelsRelativePath);

            // Load file
            string wholeFile = System.IO.File.ReadAllText(trainImagesAbsPath);

            // Split into lines
            wholeFile = wholeFile.Replace('\n', '\r');
            string[] lines = wholeFile.Split(new char[] { '\r' },
                StringSplitOptions.RemoveEmptyEntries);

            // Check number of cols and rows
            int numRows = lines.Length;
            int numCols = lines[0].Split(',').Length;

            // Allocate the data array
            string[,] values = train ? trainValues : testValues;
            

            // Load features
            for (int r = 0; r < numRows; r++)
            {
                string[] line_r = lines[r].Split(',');
                for (int c = 0; c < numCols; c++)
                {
                    values[r, c] = line_r[c];
                }
            }
        }

        public override Tuple<double[], double[]> Load(int itemIndex, bool train)
        {
            // Set values according to flag, training or testing
            string[,] values = train ? trainValues : testValues;

            // Number of features
            int length = values.GetLength(1) - 1;

            // Load features
            double[] labels;
            string temp;
            double[] data = new double[length];
            for (int i = 0; i < length; i++)
            {
                temp = values[itemIndex, i].Replace('.', ',');
                data[i] = Double.Parse(temp);
            }

            // Convert labels into categorical matrix
            if (categorical)
            {
                labels = new double[CLASSES_COUNT];
                labels[Int32.Parse(values[itemIndex, length])] = 1;
            }
            else
            {
                labels = new double[1];
                labels[0] = Int32.Parse(values[itemIndex, length]);
            }

            return new Tuple<double[], double[]>(data, labels);
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
    }
}