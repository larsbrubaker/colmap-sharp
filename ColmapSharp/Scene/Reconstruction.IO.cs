// Copyright (c) 2026, Lars Brubaker. MIT licensed (see LICENSE).
// Ported from COLMAP (BSD-3-Clause, see THIRD_PARTY_NOTICES.md).
//
// Reconstruction.IO: Reconstruction::Read, Write, ReadText, ReadBinary, WriteText and
// WriteBinary from colmap/scene/reconstruction.cc, which read and write a model directory
// (rigs, cameras, frames, images, points3D) through ReconstructionIOBinary.cs and
// ReconstructionIOText.cs. rigs and frames files are optional when reading, for models
// written before COLMAP had rigs and frames. Storage and bookkeeping are in
// Reconstruction.cs. Tests: ColmapSharp.Tests/Scene/ReconstructionIOOracleTests.cs.

using ColmapSharp.Util;

namespace ColmapSharp.Scene;

public sealed partial class Reconstruction
{
	/// <summary>
	/// Reads a model directory, binary if cameras.bin, images.bin and points3D.bin exist,
	/// else text if the .txt files do; throws if neither is complete.
	/// </summary>
	public void Read(string path)
	{
		if (File.Exists(Path.Combine(path, "cameras.bin")) && File.Exists(Path.Combine(path, "images.bin")) &&
			File.Exists(Path.Combine(path, "points3D.bin")))
		{
			ReadBinary(path);
		}
		else if (File.Exists(Path.Combine(path, "cameras.txt")) && File.Exists(Path.Combine(path, "images.txt")) &&
			File.Exists(Path.Combine(path, "points3D.txt")))
		{
			ReadText(path);
		}
		else
		{
			throw new InvalidOperationException($"rigs, cameras, frames, images, points3D files do not exist at \"{path}\"");
		}
	}

	/// <summary>Writes the model to a directory in binary format (<see cref="WriteBinary"/>).</summary>
	public void Write(string path) => WriteBinary(path);

	/// <summary>
	/// Reads a text model directory, replacing the rigs, cameras, frames, images and 3D
	/// points. As in COLMAP, the registered-frame list is not cleared.
	/// </summary>
	public void ReadText(string path)
	{
		ClearForRead();
		ReconstructionIOText.ReadCamerasText(this, Path.Combine(path, "cameras.txt"));
		string rigsPath = Path.Combine(path, "rigs.txt");
		if (File.Exists(rigsPath))
		{
			ReconstructionIOText.ReadRigsText(this, rigsPath);
		}

		string framesPath = Path.Combine(path, "frames.txt");
		if (File.Exists(framesPath))
		{
			ReconstructionIOText.ReadFramesText(this, framesPath);
		}

		ReconstructionIOText.ReadImagesText(this, Path.Combine(path, "images.txt"));
		ReconstructionIOText.ReadPoints3DText(this, Path.Combine(path, "points3D.txt"));
	}

	/// <summary>
	/// Reads a binary model directory, replacing the rigs, cameras, frames, images and 3D
	/// points. As in COLMAP, the registered-frame list is not cleared.
	/// </summary>
	public void ReadBinary(string path)
	{
		ClearForRead();
		ReconstructionIOBinary.ReadCamerasBinary(this, Path.Combine(path, "cameras.bin"));
		string rigsPath = Path.Combine(path, "rigs.bin");
		if (File.Exists(rigsPath))
		{
			ReconstructionIOBinary.ReadRigsBinary(this, rigsPath);
		}

		string framesPath = Path.Combine(path, "frames.bin");
		if (File.Exists(framesPath))
		{
			ReconstructionIOBinary.ReadFramesBinary(this, framesPath);
		}

		ReconstructionIOBinary.ReadImagesBinary(this, Path.Combine(path, "images.bin"));
		ReconstructionIOBinary.ReadPoints3DBinary(this, Path.Combine(path, "points3D.bin"));
	}

	/// <summary>Writes rigs.txt, cameras.txt, frames.txt, images.txt and points3D.txt into an existing directory.</summary>
	public void WriteText(string path)
	{
		ReconstructionIOUtils.CheckDirExists(path);
		ReconstructionIOText.WriteRigsText(this, Path.Combine(path, "rigs.txt"));
		ReconstructionIOText.WriteCamerasText(this, Path.Combine(path, "cameras.txt"));
		ReconstructionIOText.WriteFramesText(this, Path.Combine(path, "frames.txt"));
		ReconstructionIOText.WriteImagesText(this, Path.Combine(path, "images.txt"));
		ReconstructionIOText.WritePoints3DText(this, Path.Combine(path, "points3D.txt"));
	}

	/// <summary>Writes rigs.bin, cameras.bin, frames.bin, images.bin and points3D.bin into an existing directory.</summary>
	public void WriteBinary(string path)
	{
		ReconstructionIOUtils.CheckDirExists(path);
		ReconstructionIOBinary.WriteRigsBinary(this, Path.Combine(path, "rigs.bin"));
		ReconstructionIOBinary.WriteCamerasBinary(this, Path.Combine(path, "cameras.bin"));
		ReconstructionIOBinary.WriteFramesBinary(this, Path.Combine(path, "frames.bin"));
		ReconstructionIOBinary.WriteImagesBinary(this, Path.Combine(path, "images.bin"));
		ReconstructionIOBinary.WritePoints3DBinary(this, Path.Combine(path, "points3D.bin"));
	}

	// COLMAP's ReadText/ReadBinary clear cameras_, rigs_, frames_, images_ and points3D_ only;
	// reg_frame_ids_, the registered-image count and max_point3D_id_ are left as they are,
	// which only matters when reading into a non-empty reconstruction.
	private void ClearForRead()
	{
		_cameras.Clear();
		_rigs.Clear();
		_frames.Clear();
		_images.Clear();
		_points3D.Clear();
	}
}
