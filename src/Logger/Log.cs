using System;
using System.IO;

namespace Zcu.Convsharp.Logger
{
    /// <summary>
    /// Strongly inspirated by https://gist.github.com/heiswayi/69ef5413c0f28b3a58d964447c275058
    /// </summary>
    public static class Log
    {

        private static string datetimeFormat;
        private static string filename;
        private static bool consoleOutput = true;

        /// <summary>
        /// Initialize a new instance of SimpleLogger class.
        /// Log file will be created automatically if not yet exists, else it can be either a fresh new file or append to the existing file.
        /// Default is create a fresh new log file.
        /// </summary>
        /// <param name="append">True to append to existing log file, False to overwrite and create new log file</param>
        static Log()
        {
            datetimeFormat = "HH:mm:ss dd-MM-yyyy";
            filename = Path.Combine(Path.GetDirectoryName(System.Reflection.Assembly.GetEntryAssembly().Location), ".log");
            WriteLine("###### CONV# 2017", false);
            WriteLine("### Logging started", true);
        }

        /// <summary>
        /// Log a debug message
        /// </summary>
        /// <param name="text">Message</param>
        public static void Debug(string text)
        {
            WriteFormattedLog(LogLevel.DEBUG, text);
        }

        /// <summary>
        /// Log an error message
        /// </summary>
        /// <param name="text">Message</param>
        public static void Error(string text)
        {
            WriteFormattedLog(LogLevel.ERROR, text);
        }

        /// <summary>
        /// Log a fatal error message
        /// </summary>
        /// <param name="text">Message</param>
        public static void Fatal(string text)
        {
            WriteFormattedLog(LogLevel.FATAL, text);
        }

        /// <summary>
        /// Log an info message
        /// </summary>
        /// <param name="text">Message</param>
        public static void Info(string text)
        {
            WriteFormattedLog(LogLevel.INFO, text);
        }

        /// <summary>
        /// Log a trace message
        /// </summary>
        /// <param name="text">Message</param>
        public static void Trace(string text)
        {
            WriteFormattedLog(LogLevel.TRACE, text);
        }

        /// <summary>
        /// Log a waning message
        /// </summary>
        /// <param name="text">Message</param>
        public static void Warning(string text)
        {
            WriteFormattedLog(LogLevel.WARNING, text);
        }

        /// <summary>
        /// Format a log message based on log level
        /// </summary>
        /// <param name="level">Log level</param>
        /// <param name="text">Log message</param>
        private static void WriteFormattedLog(LogLevel level, string text)
        {
            string pretext;
            switch (level)
            {
                case LogLevel.TRACE: pretext = DateTime.Now.ToString(datetimeFormat) + " [TRACE] "; break;
                case LogLevel.INFO: pretext = DateTime.Now.ToString(datetimeFormat) + " [INFO] "; break;
                case LogLevel.DEBUG: pretext = DateTime.Now.ToString(datetimeFormat) + " [DEBUG] "; break;
                case LogLevel.WARNING: pretext = DateTime.Now.ToString(datetimeFormat) + " [WARNING] "; break;
                case LogLevel.ERROR: pretext = DateTime.Now.ToString(datetimeFormat) + " [ERROR] "; break;
                case LogLevel.FATAL: pretext = DateTime.Now.ToString(datetimeFormat) + " [FATAL] "; break;
                default: pretext = ""; break;
            }

            WriteLine(pretext + text);
        }

        /// <summary>
        /// Write a line of formatted log message into a log file
        /// </summary>
        /// <param name="text">Formatted log message</param>
        /// <param name="append">True to append, False to overwrite the file</param>
        /// <exception cref="System.IO.IOException"></exception>
        private static void WriteLine(string text, bool append = true)
        {
            try
            {
                using (StreamWriter outputFile = append ? File.AppendText(filename) : File.CreateText(filename))
                {
                    if (text != "")
                    {
                        // output into file
                        outputFile.WriteLine(text);
                        // output into console
                        if (consoleOutput)
                            Console.WriteLine(text);
                    }
                }
            }
            catch
            {
                throw;
            }
        }

        /// <summary>
        /// Set console output according to parameter
        /// </summary>
        /// <param name="set">if is set parameter true console will be
        /// turn on otherwise won't</param>
        public static void ConsoleOutput(bool set)
        {
            consoleOutput = set;
        }

        /// <summary>
        /// Supported log level
        /// </summary>
        [Flags]
        private enum LogLevel
        {
            TRACE,
            INFO,
            DEBUG,
            WARNING,
            ERROR,
            FATAL
        }
    }
}