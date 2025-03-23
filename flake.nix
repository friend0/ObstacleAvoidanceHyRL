{
  description = "a nix-flake-based python development environment";

  inputs.nixpkgs.url = "https://flakehub.com/f/nixos/nixpkgs/0.1.*.tar.gz";

  outputs = { self, nixpkgs }:
    let
      supportedsystems = [ "x86_64-linux" "aarch64-linux" "x86_64-darwin" "aarch64-darwin" ];
      foreachsupportedsystem = f: nixpkgs.lib.genattrs supportedsystems (system: f {
        pkgs = import nixpkgs { inherit system; };
      });
    in
    {
      devshells = foreachsupportedsystem ({ pkgs }: {
        default = pkgs.mkshell {
          packages = with pkgs; [ python313 ] ++
            (with pkgs.python313packages; [
              uv
              pkgs.zsh
              pkgs.black
              pkgs.python3
              pkgs.protobuf
            ]);

          shellhook = ''
            echo "🔧 setting up python virtual environment with uv..."

            # create venv if it doesn't exist
            if [ ! -d ".venv" ]; then
              echo "📦 no .venv found, creating with uv..."
              uv venv
            fi

            # activate the venv
            if [ -f ".venv/bin/activate" ]; then
              source .venv/bin/activate
              echo "✅ activated python venv at .venv"
              python --version
            else
              echo "❌ failed to activate venv: .venv/bin/activate not found"
            fi
          '';
        };
      });
    };
}
