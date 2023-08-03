using StaticArrays
using HDF5
using Metal
using KernelAbstractions
using Adapt

const TRIANGLE_CASES::Array{Int32, 2} = [-1 -1 -1 -1 -1 -1 -1;
3 0 2 -1 -1 -1 -1;
1 0 4 -1 -1 -1 -1;
2 3 4 2 4 1 -1;
2 1 5 -1 -1 -1 -1;
5 3 1 1 3 0 -1;
2 0 5 5 0 4 -1;
5 3 4 -1 -1 -1 -1;
4 3 5 -1 -1 -1 -1;
4 0 5 5 0 2 -1;
5 0 3 1 0 5 -1;
2 5 1 -1 -1 -1 -1;
4 3 1 1 3 2 -1;
4 0 1 -1 -1 -1 -1;
2 0 3 -1 -1 -1 -1;
-1 -1 -1 -1 -1 -1 -1] .+ 1

const EDGES::Array{Int32, 2} = [
0 1 ;
1 2 ;
2 0 ;
0 3 ;
1 3 ;
2 3 ] .+ 1

struct UnstructuredGrid{T1, T2}
  connectivity :: T1
  offsets :: T1
  ncells :: Int32
  points :: T2
end

function Adapt.adapt_structure(to, from::UnstructuredGrid)
  connectivity = adapt(to, from.connectivity)
  offsets = adapt(to, from.offsets)
  points = adapt(to, from.points)
  UnstructuredGrid(connectivity, offsets, from.ncells, points)
end

struct PermutedValues
  ids
  array
  ncomponents
end

@inline function Base.getindex(c::PermutedValues, i, j)
  idx = c.ids[i] - 1
  idx2 = idx * c.ncomponents + j
  @inbounds c.array[idx2]
end

@inline Base.getindex(c::PermutedValues, i) = Base.getindex(c, i, 1)

struct Cell
  grid
  idx
end

@inline function Base.getindex(c::Cell, i)
  @inbounds c.grid.connectivity[c.grid.offsets[c.idx]+i-1]
end

@inline function ComputeIndex(scalars, isovalue)::Int32
  CASE_MASK = SVector{4, Int32}(1, 2, 4, 8)

  index::Int32 = 0
  @inbounds for i in 1:4
    if scalars[i] >= isovalue
      index |= CASE_MASK[i]
    end
  end
  return index + 1
end

@inline function GetCell(grid, idx)
  return Cell(grid, idx)
end

@inline function GetCellValues(cell, var)
  return PermutedValues(cell, var, 1)
end

@inline function GetCellPoints(cell, var)
  return PermutedValues(cell, var, 3)
end

@kernel function CountTriangles(ntrisOut, contourValue, grid, data, triangle_cases)
  gindex = @index(Global)
  stride = @ndrange()[1]

  @inbounds for cellIdx in gindex:stride:grid.ncells
    cell = GetCell(grid, cellIdx)
    cellValues = GetCellValues(cell, data)
    ntris::UInt32 = 0
    idx = 1
    tidx = ComputeIndex(cellValues, contourValue)
    while triangle_cases[tidx, idx] > 0
      ntris += 1
      idx += 3
    end
    ntrisOut[cellIdx] = ntris
  end
end

@kernel function ContourCells(outPts, contourValue, grid, data, ntris, triOffsets, triangle_cases, edges)
  gindex = @index(Global)
  stride = @ndrange()[1]

  for cellIdx in gindex:stride:grid.ncells
    if ntris[cellIdx] < 1
      continue
    end

    cell = GetCell(grid, cellIdx)
    cellValues = GetCellValues(cell, data)
    cellPoints = GetCellPoints(cell, grid.points)
    idx = 1
    ptOffset = (triOffsets[cellIdx])*9
    ipt = 0
    tidx = ComputeIndex(cellValues, contourValue)
    while triangle_cases[tidx, idx] > 0
      for i in idx:idx+2
        eidx = triangle_cases[tidx, i]
        vs1 = edges[eidx, 1]
        vs2 = edges[eidx, 2]

        # linear interpolation across edge
        @inbounds deltaScalar::Float32 = cellValues[vs2] - cellValues[vs1]
        if deltaScalar > 0
          v1 = vs1
          v2 = vs2
        else
          v1 = vs2
          v2 = vs1
          deltaScalar = -deltaScalar
        end
        # linear interpolation across edge
        t = deltaScalar == 0.0 ? 0.0f0 : (contourValue - cellValues[v1]) / deltaScalar
        for j = 1:3
          @inbounds outPts[ptOffset+3*ipt+j] = cellPoints[v1, j] + t * (cellPoints[v2, j]-cellPoints[v1, j])
        end
        ipt += 1
      end
      idx += 3
    end
  end
end

function ReadGrid(fname::AbstractString)
  fid = h5open(fname, "r")  
  offs_::Vector{Int32} = read(fid["offsets"])
  pts::Vector{Float32} = read(fid["points"])
  data_::Vector{Float32} = read(fid["data"])
  data = adapt(backend, data_)
  conn_::Vector{Int32} = read(fid["connectivity"])
  ncells::Int32 = length(conn_)/4

  UnstructuredGrid(conn_, offs_, ncells, pts), data
end

function Contour(grid, data, contourValue)

  backend = get_backend(data)

  
  triangle_cases = adapt(backend, TRIANGLE_CASES)
  edges = adapt(backend, EDGES)

  println(1)
  ntrisOut = KernelAbstractions.allocate(backend, Int32, Int64(grid.ncells))
  @time begin
  CountTriangles(backend, 1)(ntrisOut, contourValue, adapt(backend, grid), data, triangle_cases, ndrange=10)
  end

#  println(ntrisOut)

  # @time begin
  # triOffsets = KernelAbstractions.zeros(backend, Int32, Int64(grid.ncells+1))
  # triOffsetsp = @view triOffsets[2:end]
  # accumulate!(+, triOffsets, ntrisOut)
  # end

  println(2)
  @time begin
  triOffsets = Vector{Int32}(undef, grid.ncells+1)
  triOffsets[1] = 0
  triOffsetsp = @view triOffsets[2:end]
  accumulate!(+, triOffsetsp, Vector(ntrisOut))

  nTotTris = triOffsets[end]
  #  outTris = KernelAbstractions.allocate(backend, Int32, nTotTris*3)
  outPts = KernelAbstractions.allocate(backend, Float32, nTotTris*3*3)
  end

  println(3)
  @time begin
  ContourCells(backend, 1)(outPts, contourValue, adapt(backend, grid), data, ntrisOut, adapt(backend, triOffsets), triangle_cases, edges, ndrange=10)
  end
    
  return Vector(outPts)
end

backend = CPU()
#grid, data = ReadGrid("tet-mid.h5")
grid, data = ReadGrid("tet-big.h5")

outPts = Contour(grid, data, 130)
@time Contour(grid, data, 130)
@time Contour(grid, data, 130)
@time Contour(grid, data, 130)

if false
  using ADIOS2
  adios = adios_init_serial()
  io = declare_io(adios, "IO")
  engine = open(io, "tris.bp", mode_write)

  atris = define_variable(io, "tris", outPts)

  put!(engine, atris, outPts)

  perform_puts!(engine)
  close(engine)
end
