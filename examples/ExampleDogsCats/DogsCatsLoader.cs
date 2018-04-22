using SixLabors.ImageSharp;
using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using Zcu.Convsharp.Common;
using Zcu.Convsharp.Loaders;

namespace Zcu.Convsharp
{
    /// <summary>
    /// Simple loader for loading pictures of dogs and cats
    /// from the prepared folders. You can find Dogs and cats
    /// dataset on Kaggle, URL: https://www.kaggle.com/c/dogs-vs-cats/data
    /// </summary>
    [Serializable]
    public class DogsCatsLoader : AbstractLoader2D
    {
        /// <summary>
        /// list of keys (cat or dog)(path to image)
        /// containing training images
        /// </summary>
        private List<KeyValuePair<string, string>> trainFilePaths;
        /// <summary>
        /// list of keys (cat or dog)(path to image)
        /// containing testing images
        /// </summary>
        private List<KeyValuePair<string, string>> testFilePaths;

        /// <summary>
        /// Constructor for creating instance of loader
        /// you should change the filepath according to
        /// your folder with images
        /// </summary>
        /// <param name="trainItemCount">number of train items which will be used</param>
        /// <param name="testItemCount">number of test items which will be used</param>
        /// <param name="batchSize">size of batch which will be loaded in each call</param>
        public DogsCatsLoader(int trainItemCount, int testItemCount, int batchSize, string trainPath, string testPath)
            : base(trainItemCount, testItemCount, batchSize)
        {
            trainFilePaths = new List<KeyValuePair<string, string>>();
            testFilePaths = new List<KeyValuePair<string, string>>();
            // fill trainFilePaths with valid paths to train images
            PrepareData(trainItemCount, trainPath, true);
            // fill trainFilePaths with valid paths to test images
            PrepareData(testItemCount, testPath, false);
        }

        /// <summary>
        /// Load filenames for images of cats and dogs and set
        /// the index for each item
        /// </summary>
        /// <param name="numberOfSamples">Number of images which we want to load</param>
        /// <param name="path">Path to folder with images</param>
        /// <param name="train"></param>
        private void PrepareData(int numberOfSamples, string path, bool train)
        {
            // we want to load balanced set for the simpler training
            int samplesInBalancedSet = numberOfSamples / 2;        
            // use test or training data according to flag
            List<KeyValuePair<string, string>> filepaths = train ? trainFilePaths : testFilePaths;

            int dogCounter = 0;
            int catCounter = 0;
            foreach (string file in Directory.EnumerateFiles(path, "*.jpg"))
            {
                if (Path.GetFileName(file).Contains("cat"))
                {
                    if (catCounter < samplesInBalancedSet)
                    {
                        filepaths.Add(new KeyValuePair<string, string>("cat", file));
                        catCounter++;
                    }
                }
                else
                {
                    filepaths.Add(new KeyValuePair<string, string>("dog", file));
                    dogCounter++;
                }

                // end if the set is balanced
                if (samplesInBalancedSet <= dogCounter)
                    break;
            }

            // shuffle data for faster convergence
            Shuffle(filepaths);
        }

        public override Tuple<double[][][], double[]> Load(int itemIndex, bool train)
        {
            // use test or training data according to flag
            List<KeyValuePair<string, string>> filePath = train ? trainFilePaths : testFilePaths;
            // multiplicator for normalization
            double multiplicator = 1d / 256d;

            double[][][] images;
            double[] labels;

            Image<Rgba32> image;
            using (FileStream stream = File.OpenRead(filePath.ElementAt(itemIndex).Value))
            {
                image = Image.Load(stream);

                images = Utils.Init3dArr(3, image.Width, image.Height);
                labels = new double[1];
                for (int j = 0; j < image.Width; j++)
                {
                    for (int k = 0; k < image.Height; k++)
                    {
                        images[0][j][k] = multiplicator * image[j, k].R;
                        images[1][j][k] = multiplicator * image[j, k].G;
                        images[2][j][k] = multiplicator * image[j, k].B;
                    }
                }
            }

            // set labels - for cat = 1, for dog = 0
            if (filePath.ElementAt(itemIndex).Key.Contains("cat"))
            {
                labels[0] = 1d;
            }
            else
            {
                labels[0] = 0d;
            }

            return Tuple.Create(images, labels);
        }

        /// <summary>
        /// Shuffle list
        /// </summary>
        /// <param name="list">list which will be shuffled</param>
        /// <returns>shuffled list</returns>
        public IList<T> Shuffle<T>(IList<T> list)
        {
            int n = list.Count;
            while (n > 1)
            {
                n--;
                int k = Utils.GetRandomInt(0, n + 1);
                T value = list[k];
                list[k] = list[n];
                list[n] = value;
            }
            return list;
        }
    }
}