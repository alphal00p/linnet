{
  description = "Linnet";

  inputs = {
    nixpkgs.url = "github:NixOS/nixpkgs/nixpkgs-unstable";

    crane = {
      url = "github:ipetkov/crane";
      # inputs.nixpkgs.follows = "nixpkgs";
    };

    fenix = {
      url = "github:nix-community/fenix";
      inputs.nixpkgs.follows = "nixpkgs";
      inputs.rust-analyzer-src.follows = "";
    };

    flake-utils.url = "github:numtide/flake-utils";

    advisory-db = {
      url = "github:rustsec/advisory-db";
      flake = false;
    };
  };

  outputs = {
    self,
    nixpkgs,
    crane,
    fenix,
    flake-utils,
    advisory-db,
    ...
  }:
    flake-utils.lib.eachDefaultSystem (system: let
      pkgs = nixpkgs.legacyPackages.${system};

      inherit (pkgs) lib;

      rustToolchain = fenix.packages.${system}.combine [
        (fenix.packages.${system}.stable.withComponents [
          "cargo"
          "rustc"
          "rust-src"
          "rustfmt"
        ])
        fenix.packages.${system}.targets.wasm32-unknown-unknown.stable.rust-std
      ];

      craneLib =
        (crane.mkLib nixpkgs.legacyPackages.${system}).overrideToolchain
        rustToolchain;
      src = craneLib.cleanCargoSource (craneLib.path ./.);

      iaiCallgrindRunner = pkgs.rustPlatform.buildRustPackage rec {
        pname = "iai-callgrind-runner";
        version = "0.14.0";

        src = pkgs.fetchFromGitHub {
          owner = "iai-callgrind";
          repo = "iai-callgrind";
          rev = "v0.14.0";
          hash = "sha256-NUFbA927Iye8DnmBWAQNiFmEen/a0931XlT+9gAQSV4=";
        };

        nativeBuildInputs = [
          pkgs.clang
        ];

        preConfigure = ''
          export LIBCLANG_PATH="${pkgs.llvmPackages.libclang.lib}/lib"
        '';

        subdir = "iai-callgrind-runner";

        cargoHash = "sha256-Fo76fAx5hvomFeWPGyJKdXhsaGtAmmoOU8CauZvu64I=";

        doCheck = false;
      };

      # Common arguments can be set here to avoid repeating them later
      commonArgs = {
        inherit src;
        strictDeps = true;
        pname = "linnet";
        version = "0.1.0";

        buildInputs =
          [
            # pkgs.mold
            pkgs.gcc
            pkgs.clang
            # Add additional build inputs here
          ]
          ++ lib.optionals pkgs.stdenv.isDarwin [
            # Additional darwin specific inputs can be set here
            pkgs.libiconv
            pkgs.gcc.cc.lib
          ];

        # Additional environment variables can be set directly
        # MY_CUSTOM_VAR = "some value";
      };

      rustToolchainLLvm = fenix.packages.${system}.combine [
        (fenix.packages.${system}.stable.withComponents [
          "cargo"
          "llvm-tools"
          "rustc"
        ])
        fenix.packages.${system}.targets.wasm32-unknown-unknown.stable.rust-std
      ];

      craneLibLLvmTools =
        craneLib.overrideToolchain
        rustToolchainLLvm;

      # Build *just* the cargo dependencies, so we can reuse
      # all of that work (e.g. via cachix) when running in CI
      cargoArtifacts = craneLib.buildDepsOnly commonArgs;

      # Build the actual crate itself, reusing the dependency
      # artifacts from above.
      my-crate = craneLib.buildPackage (commonArgs
        // {
          inherit cargoArtifacts;
          pname = "linnet";
          version = "0.1.0";
        });
    in {
      checks = {
        # Build the crate as part of `nix flake check` for convenience
        inherit my-crate;

        # Run clippy (and deny all warnings) on the crate source,
        # again, reusing the dependency artifacts from above.
        #
        # Note that this is done as a separate derivation so that
        # we can block the CI if there are issues here, but not
        # prevent downstream consumers from building our crate by itself.
        my-crate-clippy = craneLib.cargoClippy (commonArgs
          // {
            inherit cargoArtifacts;
            cargoClippyExtraArgs = "--all-targets -- --deny warnings";
          });

        my-crate-doc = craneLib.cargoDoc (commonArgs
          // {
            inherit cargoArtifacts;
          });

        # Check formatting
        my-crate-fmt = craneLib.cargoFmt {
          inherit src;
        };

        # Audit dependencies
        # my-crate-audit = craneLib.cargoAudit {
        #   inherit src advisory-db;
        # };

        # # Audit licenses
        # my-crate-deny = craneLib.cargoDeny {
        #   inherit src;
        # };

        # Run tests with cargo-nextest
        # Consider setting `doCheck = false` on `my-crate` if you do not want
        # the tests to run twice
        my-crate-nextest = craneLib.cargoNextest (commonArgs
          // {
            inherit cargoArtifacts;
            partitions = 1;
            partitionType = "count";
          });
      };

      packages =
        {
          default = my-crate;
        }
        // lib.optionalAttrs (!pkgs.stdenv.isDarwin) {
          my-crate-llvm-coverage = craneLibLLvmTools.cargoLlvmCov (commonArgs
            // {
              inherit cargoArtifacts;
            });
        };

      apps.default = flake-utils.lib.mkApp {
        drv = my-crate;
      };

      devShells.default = craneLib.devShell {
        # Inherit inputs from checks.
        checks = self.checks.${system};

        # Additional dev-shell environment variables can be set directly
        # MY_CUSTOM_DEVELOPMENT_VAR = "something else";
        RUST_SRC_PATH = "${pkgs.rustPlatform.rustLibSrc}";

        LD_LIBRARY_PATH = "${pkgs.stdenv.cc.cc.lib}/lib";

        EDITOR = "zed --wait";
        # Extra inputs can be added here; cargo and rustc are provided by default.
        packages = with pkgs; [
          typst
          typst-fmt
          #
          # iaiCallgrindRunner
          cargo-udeps
          cargo-insta
          cargo-deny
          cargo-edit
          cargo-watch
          rust-analyzer
          tinymist
          # nixd
          # nil
        ];
      };
    });
}
