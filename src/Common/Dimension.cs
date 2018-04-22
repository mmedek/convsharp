using System;

namespace Zcu.Convsharp.Common
{
    /// <summary>
    /// Class which is used as the crate for
    /// the three-values dimension of image
    /// </summary>
    [Serializable]
    public class Dimension
    {
        /// <summary>
        /// First value representing count of images
        /// in data
        /// </summary>
        public int imageCount;
        /// <summary>
        /// Dimension value representing WIDTH
        /// </summary>
        public int width;
        /// <summary>
        /// Dimension value representing HEIGHT
        /// </summary>
        public int height;
        /// <summary>
        /// Dimension value representing DEPTH
        /// </summary>
        public int depth;

        public Dimension()
        {
        }

        /// <summary>
        /// Constructor for the creating dimension
        /// </summary>
        /// <param name="width">width of image</param>
        /// <param name="height">height of image</param>
        /// <param name="depth">depth of image</param>
        /// <param name="imageCount">number of image</param>
        public Dimension(int imageCount, int depth, int width, int height)
        {
            this.imageCount = imageCount;
            this.width = width;
            this.height = height;
            this.depth = depth;
        }

        /// <summary>
        /// Check if are dimension values same as out dimension
        /// </summary>
        /// <param name="width">width of image</param>
        /// <param name="height">height of image</param>
        /// <param name="depth">depth of image</param>
        /// <param name="imageCount">number of image</param>
        /// <returns>If is dimension same return true otherwise false</returns>
        public bool IsSame(int imageCount, int depth, int width, int height)
        {
            return this.width == width 
                && this.height == height
                && this.depth == depth
                && this.imageCount == imageCount;
        }
    }
}