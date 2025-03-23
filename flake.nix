{
  description = "A Nix-flake-based Python development environment";

  inputs.nixpkgs.url = "https://flakehub.com/f/NixOS/nixpkgs/0.1.*.tar.gz";

  outputs = { self, nixpkgs }:
    let
      supportedSystems = [ "x86_64-linux" "aarch64-linux" "x86_64-darwin" "aarch64-darwin" ];
      forEachSupportedSystem = f: nixpkgs.lib.genAttrs supportedSystems (system: f {
        pkgs = import nixpkgs { inherit system; };
      });
    in
    {
      devShells = forEachSupportedSystem ({ pkgs }: {
        default = pkgs.mkShell {
          packages = with pkgs; [ python313 ] ++
            (with pkgs.python313Packages; [
              uv
              pkgs.zsh
              pkgs.neovim
              pkgs.black
              pkgs.grpcurl
              pkgs.python3
              pkgs.protobuf
            ]++
            [ pkgs.zsh pkgs.neovim pkgs.black pkgs.grpcurl pkgs.protobuf pkgs.texlive.combined.scheme-basic ]);

          shellHook = ''
            echo "🔧 Setting up Python virtual environment with uv..."

            # Create venv if it doesn't exist
            if [ ! -d ".venv" ]; then
              echo "📦 No .venv found, creating with uv..."
              uv venv
            fi

            # Activate the venv
            if [ -f ".venv/bin/activate" ]; then
              source .venv/bin/activate
              echo "✅ Activated Python venv at .venv"
              python --version
            else
              echo "❌ Failed to activate venv: .venv/bin/activate not found"
            fi

            if [ "$SHELL" != "$(command -v zsh)" ]; then
              export SHELL="$(command -v zsh)"
              exec zsh
            fi
          '';

      # devShells.dev = with pkgs; mkShell {
      #   buildInputs = [
      #     cargo
      #     clippy
      #     rustc
      #     rustfmt
      #     pre-commit
      #     rerun
      #     protobuf
      #   ];
      #   shellHook = ''
      #     export GIT_CONFIG=$PWD/.gitconfig
      #     export CARGO_NET_GIT_FETCH_WITH_CLI=true
      #     export GIT_SSH_COMMAND="ssh -F ~/.ssh/config"
      #     ${if pkgs.stdenv.isLinux then ''
      #       export PKG_CONFIG_PATH="${pkgs.systemd}/lib/pkgconfig:$PKG_CONFIG_PATH"
      #     '' else ""}
      #
      #     ${if pkgs.stdenv.isDarwin then ''
      #       echo "Running on macOS, using Darwin-specific dependencies."
      #     '' else ""}
      #
      #     echo "Entering Rust development environment..."
      #     cargo fetch # Pre-fetch dependencies
      #
      #   '';
      # };
        };
      });
    };
}
