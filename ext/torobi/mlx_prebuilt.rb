# frozen_string_literal: true

require "digest"
require "fileutils"
require "net/http"

# The MLX binary this extension links, fetched once and checked.
#
# MLX's Metal kernels can only be compiled by Apple's Metal toolchain,
# which is not on every machine (docs/vendoring.md). So a build without it
# takes a pre-built MLX instead, and `mlx-sys`'s build script will fetch
# one on its own if nobody has: `curl -L -f` and no check of what came
# back beyond TLS, during `gem install`, on the machine of whoever is
# installing.
#
# This does that fetch first, and refuses bytes that are not the ones this
# project was built and tested against. Then it hands the directory to
# cargo through `MLX_PREBUILT_PATH`, which is the first branch of that
# build script and the one that downloads nothing.
#
# **The digest is the point.** A release asset can be replaced, and
# nothing about the URL would change; a recorded hash is what turns "what
# GitHub serves today" into "the bytes this project ran its tests on".
#
# It is deliberately small and free of Torobi: it runs before anything is
# built, so it may use only what ships with Ruby.
module MlxPrebuilt
  # Where the binary comes from. One constant per thing that would change
  # if it came from somewhere else, so moving to another build is an edit
  # here and nowhere else.
  REPO = "takahashim/mlx-prebuilt"
  RELEASE = "v0.4.1.0"
  ASSET = "mlx-v0.30.1-mlxc-v0.4.1-macos-arm64.tar.gz"
  URL = "https://github.com/#{REPO}/releases/download/#{RELEASE}/#{ASSET}"

  # SHA-256 of that asset, checked against the bytes rather than taken
  # from the release page.
  DIGEST = "ee7af2c6d511f82af67502017d55019abb4fa2252105dd7aa2f10156f933851a"

  # What is inside, which the archive states rather than leaving to be
  # read out of a binary with `strings`. mlx-c v0.4.1 is the version whose
  # headers these bindings were generated from, and MLX v0.30.1 is what
  # that mlx-c pins: the pair as upstream tagged it.
  MLX_VERSION = "0.30.1"
  MLX_C_VERSION = "0.4.1"

  # What `mlx-sys` expects to find in a prebuilt directory.
  FILES = %w[libmlx.a libmlxc.a libgguflib.a mlx.metallib].freeze

  # A note that these four were extracted from an archive with the right
  # digest. Cheaper than hashing 140 MB on every build, and enough: what
  # it defends against is a different download, not a hostile filesystem.
  STAMP = ".verified"

  class Refused < StandardError; end

  module_function

  # The directory to hand cargo, fetching and checking it if it is not
  # already there. `MLX_PREBUILT_PATH` from the caller wins and is left
  # alone: someone who has built MLX themselves has said so.
  def ensure!(into: cache_dir, io: $stderr)
    given = ENV["MLX_PREBUILT_PATH"]
    return given if given && !given.empty?

    return libraries(into) if ready?(into)

    io.puts "torobi: fetching MLX #{MLX_VERSION} (mlx-c #{MLX_C_VERSION}) from #{URL}"
    archive = download(io:)
    verify(archive)
    unpack(archive, into)
    File.write(File.join(into, STAMP), DIGEST)
    io.puts "torobi: MLX ready in #{into}"
    libraries(into)
  ensure
    FileUtils.rm_f(archive.to_s) if archive
  end

  # Which directory of the unpacked archive holds the four.
  #
  # Searched rather than named, because the shape is the archive's
  # business: four files at the root, or an install prefix whose `lib` is
  # cmake's (`mlx/lib`). Whoever builds the archive should be able to
  # change that without this having to be told, so the only thing said
  # here is what is being looked for.
  def libraries(dir)
    found = Dir.glob(File.join(dir, "**", FILES.first))
               .map { |path| File.dirname(path) }
               .find { |place| FILES.all? { |name| File.size?(File.join(place, name)) } }
    raise Refused, "#{dir} holds no directory with #{FILES.join(", ")}" unless found

    found
  end

  # Whether a previous run left a complete, checked copy here.
  def ready?(dir)
    return false unless File.read(File.join(dir, STAMP)).strip == DIGEST

    libraries(dir)
    true
  rescue SystemCallError, Refused
    false
  end

  # One copy per machine rather than one per build directory: the
  # extension and the engine are two cargo builds of the same MLX, and
  # this is 140 MB of it.
  def cache_dir
    home = ENV["XDG_CACHE_HOME"] || File.join(Dir.home, ".cache")
    File.join(home, "torobi", "mlx-prebuilt", RELEASE)
  end

  # Streamed to disk as it arrives, and told about as it goes: this is
  # forty megabytes over whatever connection the machine has, in the
  # middle of an install, and silence for ten minutes looks like a hang.
  #
  # `Net::HTTP` rather than open-uri, which buffers the whole body into a
  # temporary file before handing it over: the same bytes written twice,
  # and no way to say how far along it is.
  def download(io:)
    require "tmpdir"
    path = File.join(Dir.mktmpdir("torobi-mlx"), ASSET)
    File.open(path, "wb") { |file| stream(URI.parse(URL), file, io) }
    path
  rescue Refused
    raise
  rescue StandardError => e
    raise Refused, "could not fetch #{URL} (#{e.class}: #{e.message}). " \
                   "Build MLX yourself and set MLX_PREBUILT_PATH to where " \
                   "#{FILES.join(", ")} are, or install the Metal toolchain " \
                   "(xcodebuild -downloadComponent MetalToolchain) and let " \
                   "MLX build from source."
  end

  # How many redirects to follow. A release asset is one hop to storage;
  # more than a few means something else is going on.
  HOPS = 5

  def stream(uri, file, io, hops: HOPS)
    raise Refused, "#{URL} redirects further than #{HOPS} hops" if hops.zero?

    Net::HTTP.start(uri.host, uri.port, use_ssl: uri.scheme == "https") do |http|
      http.request(Net::HTTP::Get.new(uri)) do |response|
        case response
        when Net::HTTPRedirection
          # Storage is a different host, and https only: a redirect that
          # drops to http is refused rather than followed.
          nxt = URI.parse(response["location"])
          raise Refused, "#{URL} redirects to #{nxt.scheme}" unless nxt.scheme == "https"

          return stream(nxt, file, io, hops: hops - 1)
        when Net::HTTPSuccess
          report = progress(response["content-length"].to_i, io)
          response.read_body do |chunk|
            file.write(chunk)
            report.call(file.size)
          end
          io.puts
        else
          raise Refused, "#{URL} answered #{response.code} #{response.message}"
        end
      end
    end
  end

  # A line that overwrites itself, or nothing at all when the size is
  # unknown or nobody is watching.
  def progress(total, io)
    return ->(_) {} if total.zero? || !io.tty?

    last = -1
    lambda do |so_far|
      percent = so_far * 100 / total
      next if percent == last

      last = percent
      io.print("\rtorobi: #{percent}% of #{total / 1024 / 1024} MB")
    end
  end

  def verify(archive)
    got = Digest::SHA256.file(archive).hexdigest
    return if got == DIGEST

    raise Refused, "#{ASSET} is not the archive this was built against.\n" \
                   "  expected #{DIGEST}\n" \
                   "  received #{got}\n" \
                   "Nothing was installed. If the release was replaced on " \
                   "purpose, the new digest belongs in " \
                   "ext/torobi/mlx_prebuilt.rb, next to the version of MLX " \
                   "it holds."
  end

  # Into a directory of its own, and only once it is whole: an unpack
  # interrupted half way should not look like a copy that is ready.
  def unpack(archive, into)
    staging = "#{into}.unpacking"
    FileUtils.rm_rf(staging)
    FileUtils.mkdir_p(staging)
    unless system("tar", "-xzf", archive, "-C", staging)
      raise Refused, "could not unpack #{archive}"
    end

    # The archive's own top directory, whatever it is called: what is
    # inside it may be the four files or a whole install prefix, and this
    # is not the place to have an opinion about which.
    top = Dir.children(staging).map { |name| File.join(staging, name) }
             .select { |path| File.directory?(path) }
    raise Refused, "#{ASSET} does not hold one directory" unless top.size == 1

    FileUtils.rm_rf(into)
    FileUtils.mkdir_p(File.dirname(into))
    FileUtils.mv(top.first, into)
    FileUtils.rm_rf(staging)
    libraries(into)
  end
end
