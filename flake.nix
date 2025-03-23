{
  description = "Development environment for ObstacleAvoidanceHyRL";

  inputs = {
    nixpkgs.url = "github:NixOS/nixpkgs/nixos-unstable";
    flake-utils.url = "github:numtide/flake-utils";
  };

  outputs = { self, nixpkgs, flake-utils }:
    flake-utils.lib.eachDefaultSystem (system:
      let
        pkgs = import nixpkgs { inherit system; };

        # Create a Python environment with the required packages.
        pythonEnv = pkgs.python3.withPackages (ps: with ps; [
          gym
          scikit-learn
          torch
          numpy
          matplotlib
          stable-baselines3
        ]);
      in {
        # Optionally, you can build the package defined by your setup.py.
        packages.default = pkgs.python3Packages.buildPythonPackage {
          pname = "hyrl";
          version = "1.0.0";
          src = ./.;
        };

        devShell = pkgs.mkShell {
          buildInputs = [
            pythonEnv
            pkgs.git
            # You might also want to add tools such as flake8, black, etc.
          ];

          shellHook = ''
            echo "Welcome to the ObstacleAvoidanceHyRL development shell!"
            echo "You can now run 'pip install -e .' to install the package in editable mode."
          '';
        };
      }
    );
}
