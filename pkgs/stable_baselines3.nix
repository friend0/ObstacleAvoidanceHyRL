{ lib, python3Packages, fetchFromGitHub }:

python3Packages.buildPythonPackage rec {
  pname = "stable-baselines3";
  version = "2.5.0";

  src = fetchFromGitHub {
    owner = "DLR-RM";
    repo = "stable-baselines3";
    rev = "656de97269c9e3051d3bbfb3f5f328d486867bd8";
    sha256 = "0zycljarks6flql38qm3pq77bnvksd6wzg4r74al43zx2dr81gcj";
  };

  propagatedBuildInputs = with python3Packages; [
    numpy
    gymnasium
    torch
    cloudpickle
    pandas
    matplotlib
  ];

  meta = with lib; {
    description = "Deep RL library based on PyTorch";
    license = licenses.mit;
    maintainers = [ "friend0" ];
  };
}
