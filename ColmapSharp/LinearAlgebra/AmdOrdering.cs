// Copyright (c) 2026, Lars Brubaker. MIT licensed (see LICENSE).
//
// AmdOrdering: the approximate minimum degree fill-reducing ordering that the sparse
// Cholesky (SimplicialCholesky.cs) applies before factoring, the replacement for
// Eigen::AMDOrdering / CHOLMOD's AMD that COLMAP's sparse solvers use by default.
//
// Written from the paper: P. R. Amestoy, T. A. Davis and I. S. Duff, "An Approximate
// Minimum Degree Ordering Algorithm", SIAM J. Matrix Anal. Appl. 17(4), 1996. No
// SuiteSparse/CSparse code was read or ported (CSparse is LGPL, docs/LICENSE_AUDIT.md), and
// the data structures here are our own (per-node lists rather than the paper's single
// in-place workspace). What is taken from the paper:
// - The quotient graph: eliminated pivots become "elements" whose variable lists L_e stand
//   for the cliques their elimination creates; each variable i keeps a list of adjacent
//   elements E_i and of still-uncovered original neighbors A_i.
// - The approximate external degree (the paper's equation for d_i): the minimum of
//   (n - k - |i|), (d_i_old + |L_p \ i|) and (|A_i \ i| + |L_p \ i| + sum over e in E_i\p of
//   |L_e \ L_p|), with |L_e \ L_p| computed for all touched elements in one pass over L_p.
// - Element absorption (every element adjacent to the pivot is absorbed) and aggressive
//   absorption (an element with |L_e \ L_p| = 0 is absorbed into the new element).
// - Supervariable detection by hashing each variable's (E_i, A_i) and comparing lists of
//   equal hash; indistinguishable variables are merged and ordered consecutively.
// Omitted as optimizations only (the result is still a valid, fill-reducing ordering):
// dense-row deferral, initial supervariable detection and mass elimination.
//
// Every weight is a count of original variables, so |x| below means the summed weights.
// Pivots are taken from the head of the lowest non-empty degree bucket; buckets are
// LIFO lists, which makes the ordering deterministic for a given input.

namespace ColmapSharp.LinearAlgebra;

/// <summary>Approximate minimum degree ordering (Amestoy, Davis and Duff 1996).</summary>
public static class AmdOrdering
{
	private enum NodeKind : byte
	{
		Variable,
		NonPrincipal,
		Element,
		AbsorbedElement,
	}

	/// <summary>
	/// Fill-reducing ordering of the symmetric pattern A + A^T of a square matrix (diagonal
	/// ignored). Returns perm with perm[k] = the original index eliminated k-th.
	/// </summary>
	public static int[] Compute(SparseMatrixCsc a) => Compute(a, out _);

	/// <summary>
	/// <see cref="Compute(SparseMatrixCsc)"/>, also returning the list entries the elimination
	/// visited: a load-independent measure of its cost that tests bound to catch quadratic
	/// behavior (a supervariable hash bucket or an element list that keeps growing).
	/// </summary>
	internal static int[] Compute(SparseMatrixCsc a, out long work)
	{
		if (a.Rows != a.Cols)
		{
			throw new ArgumentException($"AMD needs a square matrix, got {a.Rows}x{a.Cols}.", nameof(a));
		}

		var adjacency = new List<int>[a.Cols];
		for (int i = 0; i < a.Cols; i++)
		{
			adjacency[i] = [];
		}

		ReadOnlySpan<int> colPtr = a.ColPtr;
		ReadOnlySpan<int> rowIdx = a.RowIndices;
		for (int j = 0; j < a.Cols; j++)
		{
			for (int p = colPtr[j]; p < colPtr[j + 1]; p++)
			{
				int i = rowIdx[p];
				if (i != j)
				{
					adjacency[i].Add(j);
					adjacency[j].Add(i);
				}
			}
		}

		return Compute(adjacency, out work);
	}

	/// <summary>
	/// Fill-reducing ordering of an undirected graph given as adjacency lists (self loops and
	/// repeated edges are tolerated; an edge listed in one direction only counts both ways).
	/// Returns perm with perm[k] = the node eliminated k-th.
	/// </summary>
	public static int[] Compute(IReadOnlyList<IReadOnlyList<int>> adjacency) => Compute(adjacency, out _);

	private static int[] Compute(IReadOnlyList<IReadOnlyList<int>> adjacency, out long work)
	{
		int n = adjacency.Count;
		var sets = new HashSet<int>[n];
		for (int i = 0; i < n; i++)
		{
			sets[i] = [];
		}

		for (int i = 0; i < n; i++)
		{
			foreach (int j in adjacency[i])
			{
				if ((uint)j >= (uint)n)
				{
					throw new ArgumentOutOfRangeException(nameof(adjacency), $"Neighbor {j} of node {i} is outside 0..{n - 1}.");
				}

				if (j != i)
				{
					sets[i].Add(j);
					sets[j].Add(i);
				}
			}
		}

		var varAdj = new List<int>[n];
		for (int i = 0; i < n; i++)
		{
			// Sorted so the result does not depend on HashSet enumeration order.
			var list = new List<int>(sets[i]);
			list.Sort();
			varAdj[i] = list;
		}

		var state = new State(varAdj);
		int[] perm = state.Run();
		work = state.Work;
		return perm;
	}

	private sealed class State
	{
		private readonly int _n;
		private readonly NodeKind[] _kind;
		private readonly int[] _weight;
		private readonly int[] _degree;
		private readonly List<int>?[] _varAdj;
		private readonly List<int>?[] _elemAdj;
		private readonly List<int>?[] _elemVars;
		private readonly int[] _elemWeight;

		// Degree buckets: doubly linked lists threaded through _next/_prev.
		private readonly int[] _head;
		private readonly int[] _next;
		private readonly int[] _prev;

		// Members of each supervariable, as a linked list from the principal variable.
		private readonly int[] _memberNext;
		private readonly int[] _memberLast;

		// Per-pivot scratch: variable marks, |L_e \ L_p| per element, list comparison marks.
		private readonly int[] _mark;
		private readonly int[] _wStamp;
		private readonly int[] _w;
		private readonly int[] _cmpMark;
		private int _stamp;
		private int _cmpStamp;

		// List entries visited, for Compute's work count.
		public long Work { get; private set; }

		public State(List<int>[] varAdj)
		{
			_n = varAdj.Length;
			_kind = new NodeKind[_n];
			_weight = new int[_n];
			_degree = new int[_n];
			_varAdj = varAdj;
			_elemAdj = new List<int>?[_n];
			_elemVars = new List<int>?[_n];
			_elemWeight = new int[_n];
			_head = new int[_n + 1];
			_next = new int[_n];
			_prev = new int[_n];
			_memberNext = new int[_n];
			_memberLast = new int[_n];
			_mark = new int[_n];
			_wStamp = new int[_n];
			_w = new int[_n];
			_cmpMark = new int[_n];
			Array.Fill(_head, -1);
			Array.Fill(_memberNext, -1);
			for (int i = 0; i < _n; i++)
			{
				_weight[i] = 1;
				_memberLast[i] = i;
				_elemAdj[i] = [];
				_degree[i] = varAdj[i].Count;
				InsertBucket(i);
			}
		}

		public int[] Run()
		{
			var perm = new int[_n];
			int ordered = 0;
			int minDegree = 0;
			var lp = new List<int>();
			var byHash = new Dictionary<long, List<int>>();
			while (ordered < _n)
			{
				while (_head[minDegree] < 0)
				{
					Work++;
					minDegree++;
				}

				int p = _head[minDegree];
				RemoveBucket(p);
				for (int m = p; m >= 0; m = _memberNext[m])
				{
					perm[ordered++] = m;
				}

				BuildPivotElement(p, lp);
				ComputeExternalElementWeights(lp);
				UpdateLists(p, lp);
				MergeSupervariables(lp, byHash);

				int lpWeight = 0;
				foreach (int i in lp)
				{
					lpWeight += _weight[i];
				}

				int remaining = _n - ordered;
				foreach (int i in lp)
				{
					int degree = ApproximateDegree(i, p, lpWeight, remaining);
					_degree[i] = degree;
					InsertBucket(i);
					minDegree = Math.Min(minDegree, degree);
				}

				_elemVars[p] = [.. lp];
				_elemWeight[p] = lpWeight;
			}

			return perm;
		}

		/// <summary>
		/// Turns pivot p into an element: L_p is the union of A_p and of L_e over its adjacent
		/// elements, which are absorbed. Leaves the principal variables of L_p in <paramref name="lp"/>,
		/// marked with the current stamp (p is marked too, so later passes skip it).
		/// </summary>
		private void BuildPivotElement(int p, List<int> lp)
		{
			lp.Clear();
			_stamp++;
			_mark[p] = _stamp;
			Work += _elemAdj[p]!.Count + _varAdj[p]!.Count;
			foreach (int e in _elemAdj[p]!)
			{
				if (_kind[e] != NodeKind.Element)
				{
					continue;
				}

				Work += _elemVars[e]!.Count;
				foreach (int v in _elemVars[e]!)
				{
					AddToPivotList(v, lp);
				}

				_kind[e] = NodeKind.AbsorbedElement;
				_elemVars[e] = null;
			}

			foreach (int v in _varAdj[p]!)
			{
				AddToPivotList(v, lp);
			}

			_kind[p] = NodeKind.Element;
			_varAdj[p] = null;
			_elemAdj[p] = null;
			foreach (int i in lp)
			{
				RemoveBucket(i);
			}
		}

		private void AddToPivotList(int v, List<int> lp)
		{
			if (_kind[v] == NodeKind.Variable && _mark[v] != _stamp)
			{
				_mark[v] = _stamp;
				lp.Add(v);
			}
		}

		/// <summary>w(e) = |L_e \ L_p| for every live element adjacent to a variable of L_p.</summary>
		private void ComputeExternalElementWeights(List<int> lp)
		{
			foreach (int i in lp)
			{
				Work += _elemAdj[i]!.Count;
				foreach (int e in _elemAdj[i]!)
				{
					if (_kind[e] != NodeKind.Element)
					{
						continue;
					}

					if (_wStamp[e] != _stamp)
					{
						_wStamp[e] = _stamp;
						_w[e] = _elemWeight[e];
					}

					_w[e] -= _weight[i];
				}
			}
		}

		/// <summary>
		/// For each i in L_p: drop absorbed elements from E_i (aggressively absorbing any
		/// element now inside L_p) and add p; drop from A_i every variable covered by p.
		/// </summary>
		private void UpdateLists(int p, List<int> lp)
		{
			foreach (int i in lp)
			{
				List<int> elems = _elemAdj[i]!;
				Work += elems.Count + _varAdj[i]!.Count;
				int kept = 0;
				for (int t = 0; t < elems.Count; t++)
				{
					int e = elems[t];
					if (_kind[e] != NodeKind.Element)
					{
						continue;
					}

					if (_w[e] == 0)
					{
						_kind[e] = NodeKind.AbsorbedElement;
						_elemVars[e] = null;
						continue;
					}

					elems[kept++] = e;
				}

				elems.RemoveRange(kept, elems.Count - kept);
				elems.Add(p);

				List<int> vars = _varAdj[i]!;
				kept = 0;
				for (int t = 0; t < vars.Count; t++)
				{
					int v = vars[t];
					if (_kind[v] == NodeKind.Variable && _mark[v] != _stamp)
					{
						vars[kept++] = v;
					}
				}

				vars.RemoveRange(kept, vars.Count - kept);
			}
		}

		/// <summary>
		/// Merges variables of L_p with identical (E_i, A_i) into one supervariable, keeping
		/// the earliest in L_p as principal. Removes the merged ones from <paramref name="lp"/>.
		/// </summary>
		private void MergeSupervariables(List<int> lp, Dictionary<long, List<int>> byHash)
		{
			byHash.Clear();
			int kept = 0;
			for (int t = 0; t < lp.Count; t++)
			{
				int i = lp[t];
				Work += _elemAdj[i]!.Count + _varAdj[i]!.Count;
				long hash = 0;
				foreach (int e in _elemAdj[i]!)
				{
					hash += e;
				}

				foreach (int v in _varAdj[i]!)
				{
					hash += v;
				}

				bool merged = false;
				if (byHash.TryGetValue(hash, out List<int>? candidates))
				{
					foreach (int j in candidates)
					{
						Work++;
						if (SameLists(i, j))
						{
							_weight[j] += _weight[i];
							_weight[i] = 0;
							_kind[i] = NodeKind.NonPrincipal;
							_memberNext[_memberLast[j]] = i;
							_memberLast[j] = _memberLast[i];
							_elemAdj[i] = null;
							_varAdj[i] = null;
							merged = true;
							break;
						}
					}
				}
				else
				{
					candidates = [];
					byHash[hash] = candidates;
				}

				if (!merged)
				{
					candidates.Add(i);
					lp[kept++] = i;
				}
			}

			lp.RemoveRange(kept, lp.Count - kept);
		}

		private bool SameLists(int i, int j)
		{
			List<int> ei = _elemAdj[i]!, ej = _elemAdj[j]!, ai = _varAdj[i]!, aj = _varAdj[j]!;
			if (ei.Count != ej.Count || ai.Count != aj.Count)
			{
				return false;
			}

			Work += ei.Count + ai.Count + ej.Count + aj.Count;

			// Element ids and variable ids share one index space, and a node is never both,
			// so one mark pass over both lists of j compares the two sets at once.
			_cmpStamp++;
			foreach (int x in ej)
			{
				_cmpMark[x] = _cmpStamp;
			}

			foreach (int x in aj)
			{
				_cmpMark[x] = _cmpStamp;
			}

			foreach (int x in ei)
			{
				if (_cmpMark[x] != _cmpStamp)
				{
					return false;
				}
			}

			foreach (int x in ai)
			{
				if (_cmpMark[x] != _cmpStamp)
				{
					return false;
				}
			}

			return true;
		}

		/// <summary>The paper's approximate external degree of variable i after eliminating p.</summary>
		private int ApproximateDegree(int i, int p, int lpWeight, int remaining)
		{
			Work += _varAdj[i]!.Count + _elemAdj[i]!.Count;
			int outsideI = lpWeight - _weight[i];
			long bound3 = outsideI;
			foreach (int v in _varAdj[i]!)
			{
				bound3 += _weight[v];
			}

			foreach (int e in _elemAdj[i]!)
			{
				if (e != p)
				{
					bound3 += _w[e];
				}
			}

			long bound2 = (long)_degree[i] + outsideI;
			long bound1 = remaining - _weight[i];
			return (int)Math.Max(0, Math.Min(bound1, Math.Min(bound2, bound3)));
		}

		private void InsertBucket(int i)
		{
			int d = _degree[i];
			_prev[i] = -1;
			_next[i] = _head[d];
			if (_head[d] >= 0)
			{
				_prev[_head[d]] = i;
			}

			_head[d] = i;
		}

		private void RemoveBucket(int i)
		{
			if (_prev[i] >= 0)
			{
				_next[_prev[i]] = _next[i];
			}
			else
			{
				_head[_degree[i]] = _next[i];
			}

			if (_next[i] >= 0)
			{
				_prev[_next[i]] = _prev[i];
			}
		}
	}
}
