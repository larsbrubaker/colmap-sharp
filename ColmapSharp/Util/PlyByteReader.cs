// Copyright (c) 2026, Lars Brubaker. MIT licensed (see LICENSE).
//
// PlyByteReader: the std::ifstream a PLY reader (Util/Ply.cs, Util/Ply.Mesh.cs) pulls from,
// where header lines (std::getline) and the binary body (istream::read) come from the same
// buffered bytes. Written here, not a port: a block buffer over the stream, and a reused
// line buffer, so a large ASCII file costs one string per line and nothing per byte.

using System.Text;

namespace ColmapSharp.Util;

/// <summary>Buffered line and block reads over one stream, for the PLY readers.</summary>
internal sealed class PlyByteReader
{
	private static readonly UTF8Encoding LenientUtf8 = new(false, false);

	private readonly Stream _stream;
	private readonly byte[] _buffer = new byte[1 << 16];
	private int _start;
	private int _end;
	private byte[] _line = new byte[256];

	public PlyByteReader(Stream stream)
	{
		_stream = stream;
	}

	/// <summary>
	/// std::getline on the raw bytes: a line ends at '\n' (a '\r' stays for StringTrim), and
	/// null means end of stream with nothing read. Bytes are decoded as UTF-8.
	/// </summary>
	public string? ReadLine()
	{
		int length = 0;
		bool any = false;
		while (true)
		{
			if (_start == _end && !Fill())
			{
				return any ? LenientUtf8.GetString(_line, 0, length) : null;
			}

			any = true;
			ReadOnlySpan<byte> available = _buffer.AsSpan(_start, _end - _start);
			int newline = available.IndexOf((byte)'\n');
			ReadOnlySpan<byte> chunk = newline < 0 ? available : available[..newline];
			if (length + chunk.Length > _line.Length)
			{
				Array.Resize(ref _line, Math.Max(_line.Length * 2, length + chunk.Length));
			}

			chunk.CopyTo(_line.AsSpan(length));
			length += chunk.Length;
			if (newline >= 0)
			{
				_start += newline + 1;
				return LenientUtf8.GetString(_line, 0, length);
			}

			_start = _end;
		}
	}

	/// <summary>istream::read(buffer, n) followed by good(): true when all bytes were read.</summary>
	public bool ReadExactly(Span<byte> destination)
	{
		while (destination.Length > 0)
		{
			if (_start == _end && !Fill())
			{
				return false;
			}

			int count = Math.Min(destination.Length, _end - _start);
			_buffer.AsSpan(_start, count).CopyTo(destination);
			_start += count;
			destination = destination[count..];
		}

		return true;
	}

	private bool Fill()
	{
		_start = 0;
		_end = _stream.Read(_buffer, 0, _buffer.Length);
		return _end > 0;
	}
}
