# frozen_string_literal: true

require "digest"
require "fileutils"
require "json"
require "net/http"

# The MLX every cargo build here links, as a system install prefix.
#
# MLX's Metal kernels can only be compiled by Apple's Metal toolchain,
# which is not on every machine (docs/vendoring.md). So a build takes a
# pre-built MLX instead. Upstream mlx-sys compiles mlx-c against whatever
# MLX it is handed through `MLX_C_USE_SYSTEM_MLX`, which has no environment
# variable of its own, so that is driven from a generated CMake toolchain
# file (extconf.rb, the Rakefile). This fetches that MLX once, checks the
# bytes against a recorded digest, and says where the prefix landed.
#
# **The digest is the point.** A release asset can be replaced, and
# nothing about the URL would change; a recorded hash is what turns "what
# GitHub serves today" into "the bytes this project ran its tests on".
#
# It is deliberately small and free of Torobi: it runs before anything is
# built, so it may use only what ships with Ruby.
module MlxPrebuilt
  # The pin, read beside this file rather than written into it.
  #
  # A digest is written by whoever built the archive, not by whoever wrote
  # this code, and a value transcribed by hand is a value that can be
  # transcribed wrongly. `rake mlx:pin` puts it there, having downloaded
  # the bytes and hashed them; this only reads it.
  PIN_PATH = File.join(__dir__, "mlx_prebuilt.json").freeze

  # The shape of an MLX install prefix: the package `find_package(MLX)`
  # reads, the headers mlx-c compiles against, and the kernels MLX loads
  # at runtime. Everything is checked, because an archive missing any of
  # them would fail later and less clearly.
  CMAKE_CONFIG = "share/cmake/MLX/MLXConfig.cmake"
  HEADERS = "include/mlx"
  METALLIB = "lib/mlx.metallib"
  LIBS = %w[lib/libmlx.a lib/libmlxc.a lib/libgguflib.a].freeze

  # What MLX compiles its CUDA kernels against at run time. Everything
  # above is in both archives, so this is also what tells a macOS one
  # unpacked on Linux from an archive that belongs there.
  JIT_HEADERS = "include/cccl"

  # A note that this directory was extracted from an archive with the
  # right digest. Cheaper than hashing 140 MB on every build, and enough:
  # what it defends against is a different download, not a hostile
  # filesystem.
  STAMP = ".verified"

  class Refused < StandardError; end

  module_function

  # Read every time rather than captured into constants: `rake mlx:pin`
  # writes this file and then fetches through the same code an install
  # uses, so a value read at load time would still be the pin it is
  # replacing - which is how a pin could never move off an old release.
  def pin = JSON.parse(File.read(PIN_PATH))
  def repo = pin.fetch("repo")
  def release = pin.fetch("release")
  def url = "https://github.com/#{repo}/releases/download/#{release}/#{asset}"

  # Which of a release's archives this machine wants. Coarse on purpose:
  # nothing here can tell which CUDA a machine has, so moving from 12 to 13
  # is an asset and digest under the same key rather than a new one.
  def platform
    case RUBY_PLATFORM
    when /\Aarm64-darwin/ then "macos-arm64"
    when /\Ax86_64-linux/ then "linux-x86_64"
    else
      raise Refused, "torobi has no pre-built MLX for #{RUBY_PLATFORM}. Build MLX " \
                     "yourself and set TOROBI_MLX_PREFIX to its install prefix."
    end
  end

  def metal? = platform.start_with?("macos")

  # The pinned archive for this machine.
  def entry
    pin.fetch("platforms").fetch(platform) do
      raise Refused, "no MLX archive is pinned for #{platform} in #{PIN_PATH}. " \
                     "Build one with takahashim/mlx-prebuilt and `rake mlx:pin`, " \
                     "or set TOROBI_MLX_PREFIX to a prefix you built yourself."
    end
  end

  def asset = entry.fetch("asset")

  # SHA-256 of that asset. The whole point of this file: a release asset
  # can be replaced without its URL changing, and this is what notices.
  def digest = entry.fetch("digest")

  # The generation mlx-sys expects, keyed the way the manifest spells it.
  # A different question from what the archive holds: the archive is what
  # was built, this is what the bindings were generated against, and a
  # mismatch would link one library's headers to another's archive. Empty
  # for a pin that predates the field.
  #
  # `mlx-c` is a commit, not the `v0.6.0` tag: mlx-c's tag pins MLX 0.31.1
  # and seven later commits pin the 0.32.2 that mlx-sys 0.6.0 was
  # generated against, and they change headers under `mlx/c/`. A manifest
  # naming the tag is the wrong build even though it says 0.6.0.
  def required = pin.fetch("requires", {})

  # The MLX install prefix to hand mlx-c, fetching and checking it if it
  # is not already there. `TOROBI_MLX_PREFIX` from the caller wins and is
  # left alone: someone who has built MLX themselves has said so.
  def ensure!(into: cache_dir, io: $stderr)
    given = ENV.fetch("TOROBI_MLX_PREFIX", nil)
    return given if given && !given.empty?

    archive = nil
    begin
      unless ready?(into)
        # The asset names the versions it holds, which is accurate even
        # while `rake mlx:pin` moves the pin (the file's own `mlx` /
        # `mlx_c` are still the outgoing ones until the manifest is read).
        io.puts "torobi: fetching #{asset} from #{url}"
        archive = download(io:)
        verify(archive)
        unpack(archive, into)
        File.write(File.join(into, STAMP), digest)
        io.puts "torobi: MLX ready in #{into}"
      end
    ensure
      FileUtils.rm_f(archive) if archive
    end

    check_version(manifest(into:))
    prefix(into)
  end

  # Which directory of what was unpacked is the install prefix.
  #
  # Searched rather than named, because the archive's top directory is the
  # archive's business; the only thing said here is what an MLX prefix is
  # made of.
  def prefix(dir)
    found = Dir.glob(File.join(dir, "**", CMAKE_CONFIG))
               .map { |path| File.dirname(path, 4) }
               .find { |place| complete?(place) }
    unless found
      raise Refused, "#{dir} holds no MLX install prefix for #{platform} " \
                     "(wants #{(wanted_dirs + wanted_files).join(", ")})"
    end

    found
  end

  def wanted_files
    metal? ? [CMAKE_CONFIG, METALLIB, *LIBS] : [CMAKE_CONFIG, *LIBS]
  end

  def wanted_dirs
    metal? ? [HEADERS] : [HEADERS, JIT_HEADERS]
  end

  def complete?(place)
    wanted_dirs.all? { |rel| File.directory?(File.join(place, rel)) } &&
      wanted_files.all? { |rel| File.file?(File.join(place, rel)) }
  end

  # Whether the archive is the generation mlx-sys was built for.
  #
  # The digest says the bytes are the ones pinned, not that they are the
  # right MLX under mlx-c's headers; only the manifest can say that, so a
  # missing manifest is refused rather than waved through.
  def check_version(said)
    return if required.empty?

    unless said
      raise Refused, "#{asset} has no MANIFEST.txt, so which MLX it holds cannot " \
                     "be checked against mlx-sys (#{required.inspect})"
    end

    required.each do |name, want|
      got = said[name].to_s.delete_prefix("v")
      # A commit may be recorded short or full; a version is exact.
      next if got == want || got.start_with?(want)

      raise Refused, "#{asset} holds #{name} #{got.empty? ? "(unstated)" : got}, " \
                     "but mlx-sys wants #{want}. Build takahashim/mlx-prebuilt " \
                     "against that generation and `rake mlx:pin` it."
    end
  end

  # What the archive says it holds, or nil when it says nothing.
  #
  # `MANIFEST.txt` is a build's own account of itself: which mlx-c, which
  # MLX, and on what. Read from here rather than from a caller, because
  # where it sits is a fact about the archive and this is the file that
  # knows about archives.
  def manifest(into: cache_dir)
    found = Dir.glob(File.join(into, "**", "MANIFEST.txt")).first
    return nil unless found

    File.read(found).lines.filter_map do |line|
      key, value = line.split(/\s+/, 2)
      [key, value.strip] if key && value
    end.to_h
  end

  # Where mlx.metallib is, for whoever has to put it beside a binary.
  def metallib(prefix)
    File.join(prefix, METALLIB)
  end

  # Where the archives mlx-c is linked against live, for the one link
  # path upstream does not emit when MLX is a system package.
  def link_dir(prefix)
    File.join(prefix, "lib")
  end

  # A CMake toolchain file that points mlx-c at this prefix.
  #
  # Upstream mlx-sys sets no variable for this, but the `cmake` crate
  # reads `CMAKE_TOOLCHAIN_FILE` from the environment, which is upstream
  # being told without being changed (docs/vendoring.md). Written rather
  # than committed: it names a path on this machine.
  #
  # It goes beside the extracted prefix in the cache, not into the
  # checkout, so a fixed path is never part of the repository.
  def toolchain_file(prefix, into: cache_dir)
    FileUtils.mkdir_p(into)
    path = File.join(into, "mlx_toolchain.cmake")
    File.write(path, <<~CMAKE)
      # Generated by ext/torobi/mlx_prebuilt.rb; do not edit or commit.
      # Found by mlx-sys through CMAKE_TOOLCHAIN_FILE, which is how a
      # pre-built MLX reaches mlx-c without a fork of mlx-sys.
      set(MLX_C_USE_SYSTEM_MLX ON CACHE BOOL "" FORCE)
      set(CMAKE_PREFIX_PATH "#{prefix.tr("\\", "/")}" CACHE STRING "" FORCE)
      # mlx-c builds its examples by default, and they link MLX's vendored
      # gguflib, which is built in-tree only when MLX is. With a system MLX
      # there is nothing to link and the build stops on example-gguf; the
      # library does not need them, so they are turned off.
      set(MLX_C_BUILD_EXAMPLES OFF CACHE BOOL "" FORCE)
    CMAKE
    path
  end

  # Whether a previous run left a complete, checked copy here.
  def ready?(dir)
    return false unless File.read(File.join(dir, STAMP)).strip == digest

    prefix(dir)
    true
  rescue SystemCallError, Refused
    false
  end

  # One copy per machine rather than one per build directory: the
  # extension and the engine are two cargo builds of the same MLX, and
  # this is 140 MB of it.
  def cache_dir
    home = ENV["XDG_CACHE_HOME"] || File.join(Dir.home, ".cache")
    File.join(home, "torobi", "mlx-prebuilt", release)
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
    path = File.join(Dir.mktmpdir("torobi-mlx"), asset)
    File.open(path, "wb") { |file| stream(URI.parse(url), file, io) }
    path
  rescue Refused
    raise
  rescue StandardError => e
    raise Refused, "could not fetch #{url} (#{e.class}: #{e.message}). " \
                   "Build MLX yourself and set TOROBI_MLX_PREFIX to its install " \
                   "prefix, or install the Metal toolchain " \
                   "(xcodebuild -downloadComponent MetalToolchain) and build MLX " \
                   "from source."
  end

  # How many redirects to follow. A release asset is one hop to storage;
  # more than a few means something else is going on.
  HOPS = 5

  def stream(uri, file, io, hops: HOPS)
    raise Refused, "#{url} redirects further than #{HOPS} hops" if hops.zero?

    Net::HTTP.start(uri.host, uri.port, use_ssl: uri.scheme == "https") do |http|
      http.request(Net::HTTP::Get.new(uri)) do |response|
        case response
        when Net::HTTPRedirection
          # Storage is a different host, and https only: a redirect that
          # drops to http is refused rather than followed.
          nxt = URI.parse(response["location"])
          raise Refused, "#{url} redirects to #{nxt.scheme}" unless nxt.scheme == "https"

          return stream(nxt, file, io, hops: hops - 1)
        when Net::HTTPSuccess
          report = progress(response["content-length"].to_i, io)
          response.read_body do |chunk|
            file.write(chunk)
            report.call(file.size)
          end
          io.puts
        else
          raise Refused, "#{url} answered #{response.code} #{response.message}"
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
    return if got == digest

    raise Refused, "#{asset} is not the archive this was built against.\n  " \
                   "expected #{digest}\n  " \
                   "received #{got}\n" \
                   "Nothing was installed. If the release was replaced on " \
                   "purpose, the new digest belongs in " \
                   "ext/torobi/mlx_prebuilt.json, next to the version of MLX " \
                   "it holds; `rake mlx:pin` puts it there."
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

    prefix(staging) # refuse a wrong archive before it is given a name
    FileUtils.rm_rf(into)
    FileUtils.mkdir_p(File.dirname(into))
    FileUtils.mv(staging, into)
    prefix(into)
  end
end
