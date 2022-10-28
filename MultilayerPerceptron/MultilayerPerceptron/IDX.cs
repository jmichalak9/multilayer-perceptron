using System.IO.Compression;

namespace MultilayerPerceptron;

public class IDX
{
    public byte[] data { get; }
    public int[] dim { get; }
    public IDX(string filepath)
    {
        var data = File.OpenRead(filepath);
        var zipStream = new GZipStream(data, CompressionMode.Decompress, true);
        var stream = new BinaryReader(zipStream);

        var magic = new byte[4];
        stream.Read(magic, 0, 4);
        if (magic[2] != 0x08)
        {
            throw new Exception();
        }

        var nDim = (int)magic[3];
        dim = new int[nDim];
        for (int i = 0; i < nDim; i++)
        {
            var bytes = stream.ReadBytes(4);
            Array.Reverse(bytes);
            dim[i] =  BitConverter.ToInt32(bytes, 0);
        }
        int totalLen = 1;
        dim[0] = 1000;
        foreach(var d in dim)
        {
            totalLen = totalLen * d;
        }
        this.data = stream.ReadBytes((int)totalLen);
        
    }
}