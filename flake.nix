{
  description = "A Nix-flake-based Python development environment";

  inputs.nixpkgs.url = "https://flakehub.com/f/NixOS/nixpkgs/0.1.*.tar.gz";

  outputs = { self, nixpkgs }:
    let
      supportedSystems = [ "x86_64-linux" "aarch64-linux" "aarch64-darwin" ];
      forEachSupportedSystem = f:
        nixpkgs.lib.genAttrs supportedSystems
        (system: f { pkgs = import nixpkgs { system = system; config.allowUnfree = true; }; });
    in {
      packages = forEachSupportedSystem ({ pkgs }: {
        default = pkgs.writeShellScriptBin "hyrl-server" ''
          export PATH="${pkgs.python310}/bin:$PATH"
          export LD_LIBRARY_PATH=${pkgs.gcc.cc.lib}/lib:$LD_LIBRARY_PATH
          
          # Create a temporary working directory
          WORK_DIR=$(mktemp -d)
          trap "rm -rf $WORK_DIR" EXIT
          
          # Copy source to writable location
          cp -r ${self}/* $WORK_DIR/
          cd $WORK_DIR
          
          # Set PYTHONPATH to include src directory
          export PYTHONPATH="$WORK_DIR/src:$PYTHONPATH"
          
          # Run the server using uv
          exec ${pkgs.python310Packages.uv}/bin/uv run python src/rl_policy/server.py
        '';
      });

      apps = forEachSupportedSystem ({ pkgs }: {
        default = {
          type = "app";
          program = "${self.packages.${pkgs.system}.default}/bin/hyrl-server";
        };
      });

      devShells = forEachSupportedSystem ({ pkgs }: {
        # Default shell for just running the code (minimal setup)
        default = pkgs.mkShell {
          packages = with pkgs; [
            python310
            python310Packages.uv
            grpcurl
            protobuf
            mypy-protobuf
          ];

          shellHook = ''
            echo "🔧 Setting up minimal Python environment with uv..."

            # Add GCC library path to LD_LIBRARY_PATH
            export LD_LIBRARY_PATH=${pkgs.gcc.cc.lib}/lib:$LD_LIBRARY_PATH

            # Add src to PATH
            export PATH="${pkgs.python310}/bin:$PATH"
            export PYTHONPATH=$PWD/src:$PYTHONPATH

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
              # Install dependencies from pyproject.toml if it exists
              if [ -f "pyproject.toml" ]; then
                echo "📦 Installing dependencies from pyproject.toml..."
                uv pip install -e .
              else
                echo "⚠️ No pyproject.toml found, skipping dependency installation"
              fi
            else
              echo "❌ Failed to activate venv: .venv/bin/activate not found"
            fi
          '';
        };

        # Dev shell geared toward development, leaving your system's Neovim in place
        dev = pkgs.mkShell {
          packages = with pkgs; [
            python310
            python310Packages.uv
            zsh
            black
            texliveFull
            grpcurl
            protobuf
            mypy-protobuf
          ];

          shellHook = ''
            echo "🔧 Setting up Python development environment with uv..."

            # Add GCC library path to LD_LIBRARY_PATH
            export LD_LIBRARY_PATH=${pkgs.gcc.cc.lib}/lib:$LD_LIBRARY_PATH

            # Add src to PATH
            export PYTHONPATH=$PWD/src:$PYTHONPATH

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
              # Install dependencies from pyproject.toml with dev extras
              if [ -f "pyproject.toml" ]; then
                echo "📦 Installing dependencies from pyproject.toml..."
                uv pip install -e .
              else
                echo "⚠️ No pyproject.toml found, skipping dependency installation"
              fi
            fi

            # Start Zsh if not already the active shell
            if [ "$SHELL" != "$(command -v zsh)" ]; then
              export SHELL="$(command -v zsh)"
              exec zsh
            fi
          '';
        };

        # Full development environment with neovim included
        full = pkgs.mkShell {
          packages = with pkgs; [
            python310
            python310Packages.uv
            zsh
            neovim
            black
            texliveFull
            grpcurl
            protobuf
            mypy-protobuf
          ];

          shellHook = ''
            echo "🔧 Setting up comprehensive Python development environment with uv..."

            # Add GCC library path to LD_LIBRARY_PATH
            export LD_LIBRARY_PATH=${pkgs.gcc.cc.lib}/lib:$LD_LIBRARY_PATH

            # Add src to PATH
            export PATH="${pkgs.python310}/bin:$PATH"
            export PYTHONPATH=$PWD/src:$PYTHONPATH

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
              # Install dependencies from pyproject.toml with dev extras
              if [ -f "pyproject.toml" ]; then
                echo "📦 Installing dependencies from pyproject.toml..."
                uv pip install -e .[dev]
              else
                echo "⚠️ No pyproject.toml found, skipping dependency installation"
              fi
            fi

            # Start Zsh if not already the active shell
            if [ "$SHELL" != "$(command -v zsh)" ]; then
              export SHELL="$(command -v zsh)"
              exec zsh
            fi
          '';
        };

        # Alias to make run the same as default
        run = self.devShells.${pkgs.system}.default;
      });
    };
}
